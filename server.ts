import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';
import { storage } from './server/storage';
import { ai, buildSystemInstruction, generateContentStreamWithFallback, generateContentWithFallback } from './server/gemini';
import { VOICE_OPTIONS } from './src/constants/companionDefaults';
import { MemoryCategory } from './src/types/companion';
import { getCacheKey, getCachedTTS, setCachedTTS } from './server/ttsCache';
import { synthesizeAcousticSpeech } from './server/acousticSynth';

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const isProd = process.env.NODE_ENV === 'production';

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Helper auth middleware
function getUserId(req: Request): string {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.substring(7);
    if (token.startsWith('companion_token_')) {
      return token.replace('companion_token_', '');
    }
  }
  return 'user_default';
}

// -------------------------------------------------------------
// AUTH ROUTES
// -------------------------------------------------------------
app.post('/api/auth/register', (req: Request, res: Response) => {
  try {
    const { email, password, name } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }
    const existing = storage.getUserByEmail(email);
    if (existing) {
      return res.status(400).json({ error: 'An account with this email already exists' });
    }
    const user = storage.createUser(email, password, name || email.split('@')[0]);
    const token = `companion_token_${user.id}`;
    res.json({ user: { id: user.id, email: user.email, name: user.name }, token });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Registration failed' });
  }
});

app.post('/api/auth/login', (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }
    const user = storage.getUserByEmail(email);
    if (!user || user.passwordHash !== password) {
      return res.status(401).json({ error: 'Invalid email or password' });
    }
    const token = `companion_token_${user.id}`;
    res.json({ user: { id: user.id, email: user.email, name: user.name }, token });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Login failed' });
  }
});

app.post('/api/auth/quick-guest', (req: Request, res: Response) => {
  try {
    const id = 'guest_' + Math.random().toString(36).substring(2, 9);
    const guestUser = storage.createUser(`${id}@companion.ai`, 'guest_pass', 'Guest Explorer');
    const token = `companion_token_${guestUser.id}`;
    res.json({ user: { id: guestUser.id, email: guestUser.email, name: guestUser.name }, token });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Guest creation failed' });
  }
});

app.get('/api/auth/me', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const user = storage.getUserById(userId);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }
  res.json({ user: { id: user.id, email: user.email, name: user.name } });
});

app.post('/api/auth/logout', (_req: Request, res: Response) => {
  res.json({ success: true });
});

app.post('/api/auth/delete-account', (req: Request, res: Response) => {
  const userId = getUserId(req);
  storage.deleteUser(userId);
  res.json({ success: true, message: 'Account and associated memories deleted permanently.' });
});

app.get('/api/auth/export', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const data = storage.exportUserData(userId);
  res.json(data);
});

// -------------------------------------------------------------
// IDENTITY ROUTES
// -------------------------------------------------------------
app.get('/api/identity', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const identity = storage.getIdentity(userId);
  res.json(identity);
});

app.post('/api/identity', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const updated = storage.updateIdentity(userId, req.body);
  res.json(updated);
});

// -------------------------------------------------------------
// CONVERSATION ROUTES
// -------------------------------------------------------------
app.get('/api/conversations', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const convs = storage.getConversations(userId);
  res.json(convs);
});

app.post('/api/conversations', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const { title } = req.body;
  const conv = storage.createConversation(userId, title);
  res.json(conv);
});

app.get('/api/conversations/:id', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const conv = storage.getConversation(userId, req.params.id);
  if (!conv) return res.status(404).json({ error: 'Conversation not found' });
  res.json(conv);
});

app.patch('/api/conversations/:id', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const updated = storage.updateConversation(userId, req.params.id, req.body);
  if (!updated) return res.status(404).json({ error: 'Conversation not found' });
  res.json(updated);
});

app.delete('/api/conversations/:id', (req: Request, res: Response) => {
  const userId = getUserId(req);
  storage.deleteConversation(userId, req.params.id);
  res.json({ success: true });
});

app.post('/api/conversations/:id/messages', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const conv = storage.getConversation(userId, req.params.id);
  if (!conv) return res.status(404).json({ error: 'Conversation not found' });

  const message = req.body;
  conv.messages.push(message);
  conv.updatedAt = new Date().toISOString();

  // Auto-generate title after 2 messages if title is still default
  if (conv.messages.length === 2 && conv.title === 'New Conversation') {
    const firstUserMsg = conv.messages.find((m) => m.role === 'user');
    if (firstUserMsg) {
      conv.title = firstUserMsg.content.slice(0, 32).trim() + (firstUserMsg.content.length > 32 ? '...' : '');
    }
  }

  storage.updateConversation(userId, conv.id, { messages: conv.messages, title: conv.title });
  res.json(conv);
});

// -------------------------------------------------------------
// MEMORY ROUTES
// -------------------------------------------------------------
app.get('/api/memories', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const memories = storage.getMemories(userId);
  res.json(memories);
});

app.post('/api/memories', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const { category, content, sourceContext, imageUrl } = req.body;
  if (!category || !content) {
    return res.status(400).json({ error: 'Category and content are required' });
  }
  const item = storage.addMemory(userId, category, content, sourceContext || 'User created', imageUrl);
  res.json(item);
});

app.put('/api/memories/:id', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const { content, category } = req.body;
  const updated = storage.updateMemory(userId, req.params.id, content, category);
  if (!updated) return res.status(404).json({ error: 'Memory not found' });
  res.json(updated);
});

app.delete('/api/memories/:id', (req: Request, res: Response) => {
  const userId = getUserId(req);
  storage.deleteMemory(userId, req.params.id);
  res.json({ success: true });
});

app.delete('/api/memories', (req: Request, res: Response) => {
  const userId = getUserId(req);
  const { category, all } = req.query;
  if (all === 'true') {
    storage.clearAllMemories(userId);
  } else if (category) {
    storage.deleteMemoriesByCategory(userId, category as MemoryCategory);
  }
  res.json({ success: true });
});

// -------------------------------------------------------------
// AI ENDPOINTS (GEMINI 3.8 FLASH & TTS)
// -------------------------------------------------------------

// SSE Streaming chat endpoint
app.post('/api/ai/chat-stream', async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);
    const { messages, visualData, isScreenSharing, isWatchTogether } = req.body;

    const identity = storage.getIdentity(userId);
    const memories = identity.memoryEnabled ? storage.getMemories(userId) : [];
    const systemInstruction = buildSystemInstruction(identity, memories, isScreenSharing, isWatchTogether);

    // Setup SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Prepare contents array for Gemini
    const contents: any[] = [];

    // Map conversation history (keep last 12 messages for fast low-latency context)
    const recent = (messages || []).slice(-12);
    for (let i = 0; i < recent.length; i++) {
      const msg = recent[i];
      const role = msg.role === 'assistant' ? 'model' : 'user';

      if (i === recent.length - 1 && visualData && visualData.base64) {
        // Last turn has an image attachment (from Screen Share or Camera)
        const mimeType = visualData.mimeType || 'image/jpeg';
        const cleanBase64 = visualData.base64.replace(/^data:image\/\w+;base64,/, '');
        contents.push({
          role: 'user',
          parts: [
            {
              inlineData: {
                mimeType,
                data: cleanBase64,
              },
            },
            {
              text: msg.content || 'Analyze what is currently visible on my screen / camera and respond naturally.',
            },
          ],
        });
      } else {
        contents.push({
          role,
          parts: [{ text: msg.content }],
        });
      }
    }

    if (contents.length === 0) {
      contents.push({
        role: 'user',
        parts: [{ text: 'Hello! Please introduce yourself.' }],
      });
    }

    // Call generateContentStream on gemini with resilient fallback
    const responseStream = await generateContentStreamWithFallback({
      contents,
      systemInstruction,
      temperature: 0.8,
      topP: 0.95,
    });

    let detectedEmotion = 'calm';
    let fullTextAccumulator = '';
    let emotionExtracted = false;

    for await (const chunk of responseStream) {
      const text = chunk.text || '';
      if (!text) continue;
      fullTextAccumulator += text;

      // Extract [EMOTION:...] tag if present in the opening tokens
      if (!emotionExtracted && fullTextAccumulator.includes('[EMOTION:')) {
        const match = fullTextAccumulator.match(/\[EMOTION:([a-z_]+)\]/i);
        if (match) {
          detectedEmotion = match[1].toLowerCase();
          emotionExtracted = true;
        }
      }

      // Stream text chunk out to client
      res.write(`data: ${JSON.stringify({ text, emotion: detectedEmotion })}\n\n`);
    }

    res.write(`data: ${JSON.stringify({ done: true, fullText: fullTextAccumulator, emotion: detectedEmotion })}\n\n`);
    res.end();
  } catch (err: any) {
    console.error('Chat stream error:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: err.message || 'AI streaming failed' });
    } else {
      res.write(`data: ${JSON.stringify({ error: err.message || 'Stream interrupted' })}\n\n`);
      res.end();
    }
  }
});

// Autonomous Memory Extractor (runs in background or on demand)
app.post('/api/ai/extract-memories', async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);
    const { userMessage, aiResponse } = req.body;
    const identity = storage.getIdentity(userId);

    if (!identity.memoryEnabled) {
      return res.json({ extracted: [] });
    }

    // Fast check: if message is tiny or a simple greeting, skip
    if (!userMessage || userMessage.trim().length < 8) {
      return res.json({ extracted: [] });
    }

    const prompt = `Analyze this conversation snippet between a user and their personal companion.
User: "${userMessage}"
AI: "${aiResponse || ''}"

Identify if the user explicitly shared any persistent personal facts, preferences, hobbies, job/project details, family/pet details, important dates, or goals worth remembering long term.
Do NOT store trivial greetings, transient questions, or generic remarks.

Return a JSON array of remembered facts. If nothing worth remembering, return an empty array.`;

    const response = await generateContentWithFallback({
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              category: {
                type: Type.STRING,
                description:
                  'One of: user_name, preferences, interests, hobbies, favorite_topics, communication_style, important_dates, projects, goals, frequently_discussed, important_instructions, user_notes',
              },
              content: {
                type: Type.STRING,
                description: 'Clear, concise factual memory statement in 3rd person about the user.',
              },
            },
            required: ['category', 'content'],
          },
        },
      },
    });

    const parsed = JSON.parse(response.text || '[]');
    const saved: any[] = [];

    if (Array.isArray(parsed)) {
      for (const item of parsed) {
        if (item.category && item.content) {
          const added = storage.addMemory(userId, item.category as MemoryCategory, item.content, userMessage);
          saved.push(added);
        }
      }
    }

    res.json({ extracted: saved });
  } catch (err: any) {
    console.error('Memory extraction error:', err);
    res.json({ extracted: [] });
  }
});

// High-Fidelity Text-to-Speech (Gemini TTS)
app.post('/api/ai/tts', async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);
    const { text, voicePresetId, pitch, speed, rate, volume } = req.body;

    if (!text || text.trim().length === 0) {
      return res.status(400).json({ error: 'Text is required for TTS' });
    }

    // Clean any emotion tags or markdown artifacts before speaking
    const cleanText = text
      .replace(/\[EMOTION:[a-z_]+\]/gi, '')
      .replace(/```[\s\S]*?```/g, 'Code block omitted for speech.')
      .replace(/[*#_`]/g, '')
      .trim();

    if (!cleanText) {
      return res.status(400).json({ error: 'No speakable text' });
    }

    // Find requested or configured voice
    const identity = storage.getIdentity(userId);
    const selectedVoiceId = voicePresetId || identity.voicePresetId;
    const voiceDef = VOICE_OPTIONS.find((v) => v.id === selectedVoiceId) || VOICE_OPTIONS[0];

    // Compute active speaker parameters
    const effectivePitch = typeof pitch === 'number' ? pitch : (identity.voicePitch || voiceDef.pitch || 1.0);
    const effectiveRate = typeof speed === 'number' ? speed : (typeof rate === 'number' ? rate : (identity.voiceSpeed || voiceDef.rate || 1.0));
    const effectiveVolume = typeof volume === 'number' ? volume : (identity.voiceVolume ?? 1.0);

    const speakerParameters = {
      voicePresetId: voiceDef.id,
      voiceName: voiceDef.name,
      geminiVoice: voiceDef.geminiVoice,
      gender: voiceDef.gender,
      pitch: Number(effectivePitch.toFixed(2)),
      speed: Number(effectiveRate.toFixed(2)),
      volume: Number(effectiveVolume.toFixed(2)),
      style: voiceDef.style,
      language: identity.language,
      arabicDialect: identity.arabicDialect,
    };

    // 1. Check persistent TTS cache first (keyed with pitch and rate)
    const paramSuffix = `p${effectivePitch.toFixed(2)}_r${effectiveRate.toFixed(2)}`;
    const cacheKey = getCacheKey(`${cleanText}_${paramSuffix}`, voiceDef.id);
    const cachedAudio = getCachedTTS(cacheKey);
    if (cachedAudio) {
      return res.json({
        audioBase64: cachedAudio,
        mimeType: 'audio/wav',
        voice: voiceDef.name,
        speakerParameters,
        cached: true,
      });
    }

    // 2. Models to try (Strictly valid models from Gemini API guidelines)
    const modelsToTry = [
      'gemini-3.8-flash-tts',
      'gemini-3.8-flash-lite-tts',
    ];

    let base64Audio: string | undefined;

    const pitchDescriptor = effectivePitch > 1.08 ? 'higher pitch, bright' : effectivePitch < 0.92 ? 'lower pitch, deep resonant' : 'balanced pitch';
    const rateDescriptor = effectiveRate > 1.08 ? 'brisk dynamic pace' : effectiveRate < 0.92 ? 'deliberate, measured pace' : 'natural conversation pace';

    for (const model of modelsToTry) {
      try {
        const response = await ai.models.generateContent({
          model,
          contents: [
            {
              role: 'user',
              parts: [
                {
                  text: cleanText.slice(0, 800),
                  speechMetadata: {
                    style: `${voiceDef.style}, ${pitchDescriptor}, ${rateDescriptor}`,
                  },
                },
              ],
            },
          ],
          config: {
            responseModalities: ['AUDIO'],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: voiceDef.geminiVoice },
              },
            },
          },
        });

        const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (data) {
          base64Audio = data;
          setCachedTTS(cacheKey, data);
          break;
        }
      } catch (err: any) {
        // Continue to fallback model if 429 quota or other transient error
      }
    }

    if (base64Audio) {
      return res.json({
        audioBase64: base64Audio,
        mimeType: 'audio/wav',
        voice: voiceDef.name,
        speakerParameters,
      });
    }

    // 3. Robust acoustic synthesized fallback with speaker parameters applied
    try {
      const synthAudio = synthesizeAcousticSpeech(cleanText, {
        gender: voiceDef.gender,
        pitch: effectivePitch,
        rate: effectiveRate,
      });

      setCachedTTS(cacheKey, synthAudio);

      return res.json({
        audioBase64: synthAudio,
        mimeType: 'audio/wav',
        voice: voiceDef.name,
        speakerParameters,
        acousticFallback: true,
      });
    } catch (synthErr) {
      console.warn('Acoustic synth fallback error:', synthErr);
    }

    res.json({
      fallback: true,
      voice: voiceDef.name,
      message: 'Using browser acoustic synthesis',
    });
  } catch (err: any) {
    console.error('TTS generation error:', err);
    res.json({ fallback: true, error: err.message || 'TTS fallback' });
  }
});

// Single Snapshot Visual Analysis (Camera / Screen)
app.post('/api/ai/analyze-visual', async (req: Request, res: Response) => {
  try {
    const userId = getUserId(req);
    const { imageBase64, prompt, mode } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: 'Image base64 required' });
    }

    const identity = storage.getIdentity(userId);
    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');

    const systemPrompt = `You are "${identity.name}", analyzing a live visual frame (${mode || 'screen'}).
Explain what you see concisely, accurately, and naturally. If it's a video/movie, identify the action/scene. If it's code/doc, identify the subject. Keep it engaging.`;

    const response = await generateContentWithFallback({
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: cleanBase64,
            },
          },
          {
            text: prompt || 'Describe what you notice here.',
          },
        ],
      },
      config: {
        systemInstruction: systemPrompt,
      },
    });

    res.json({ analysis: response.text });
  } catch (err: any) {
    console.error('Visual analysis error:', err);
    res.status(500).json({ error: err.message || 'Visual analysis failed' });
  }
});

// -------------------------------------------------------------
// VITE OR STATIC ASSETS
// -------------------------------------------------------------
async function startServer() {
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[Aura Companion Server] Running on http://0.0.0.0:${PORT} (${isProd ? 'production' : 'development'})`);
  });
}

startServer();
