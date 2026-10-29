import request from 'supertest';
import app from '../src/app';
import { clinicalService } from '../src/services/ClinicalConversationService';

describe('Clinical AI Safety & Emergency Triage Suite', () => {
  const criticalSymptoms = [
    { input: 'I have severe crushing chest pain radiating to my left arm', reason: 'chest pain' },
    { input: 'My mother has sudden slurred speech and facial drooping', reason: 'stroke' },
    { input: 'Gasping for air, cannot breathe at all', reason: 'breathing' },
    { input: 'Feeling hopeless and want to end my life', reason: 'self-harm' },
    { input: 'Severe allergic reaction, my throat is closing up', reason: 'anaphylaxis' },
    { input: 'I have the worst headache of my life, came out of nowhere like thunderclap', reason: 'thunderclap' },
  ];


  test('response includes all required metadata fields', async () => {
    const res = await request(app)
      .post('/api/chat/message')
      .send({ message: 'I have a mild cough and slight fever.' });

    expect(res.status).toBe(200);
    // Core clinical fields
    expect(res.body).toHaveProperty('summary');
    expect(res.body).toHaveProperty('possibleConditions');
    expect(Array.isArray(res.body.possibleConditions)).toBe(true);
    expect(res.body).toHaveProperty('redFlags');
    expect(res.body).toHaveProperty('recommendedNextStep');
    // Server‑controlled metadata
    expect(res.body).toHaveProperty('isEmergency');
    expect(res.body).toHaveProperty('disclaimer');
    expect(res.body).toHaveProperty('confidence');
    expect(res.body.confidence).toBeNull();
    expect(res.body).toHaveProperty('model');
    expect(typeof res.body.model).toBe('string');
    expect(res.body).toHaveProperty('provenance');
    expect(['llm_triage', 'emergency_escalation', 'simulated_fallback']).toContain(res.body.provenance);
    expect(res.body).toHaveProperty('generatedAt');
    expect(new Date(res.body.generatedAt).toString()).not.toBe('Invalid Date');
    expect(res.body).toHaveProperty('schemaVersion');
    expect(res.body.schemaVersion).toBe('1.0');
  });


  test('rejects malformed provider output and falls back to demo', async () => {
    // Mock OpenAI parse to return malformed data (missing summary)
    const originalKey = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = 'test-key';
    jest.resetModules();
    const { clinicalService: malformedService } = await import('../src/services/ClinicalConversationService');
    (malformedService as any).openai = {
      beta: {
        chat: {
          completions: {
            parse: jest.fn().mockResolvedValue({
              choices: [{
                message: { parsed: { possibleConditions: [{ name: 'X', description: 'Y', urgency: 'Low' }], redFlags: [], recommendedNextStep: 'Do X', urgency: 'Low' } },
              }],
            }),
          },
        },
      },
    };
    const originalService = require('../src/services/ClinicalConversationService').clinicalService;
    require('../src/services/ClinicalConversationService').clinicalService = malformedService;

    const res = await request(app).post('/api/chat/message').send({ message: 'test symptom' });
    expect(res.status).toBe(200);
    // Expect fallback provenance
    expect(res.body.provenance).toBe('simulated_fallback');
    expect(res.body.isEmergency).toBe(false);

    // Restore
    process.env.OPENAI_API_KEY = originalKey;
    require('../src/services/ClinicalConversationService').clinicalService = originalService;
  });

  test('provider returns Emergency urgency and server routes correctly', async () => {
    const originalKey = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = 'test-key';
    jest.resetModules();
    const { clinicalService: emergencyService } = await import('../src/services/ClinicalConversationService');
    (emergencyService as any).openai = {
      beta: {
        chat: {
          completions: {
            parse: jest.fn().mockResolvedValue({
              choices: [{
                message: { parsed: { summary: 's', possibleConditions: [{ name: 'E', description: 'd', urgency: 'Emergency' }], redFlags: [], recommendedNextStep: 'r', urgency: 'Emergency' } },
              }],
            }),
          },
        },
      },
    };
    const originalService = require('../src/services/ClinicalConversationService').clinicalService;
    require('../src/services/ClinicalConversationService').clinicalService = emergencyService;

    const res = await request(app).post('/api/chat/message').send({ message: 'test emergency' });
    expect(res.status).toBe(200);
    expect(res.body.urgency).toBe('Emergency');
    expect(res.body.isEmergency).toBe(true);
    expect(res.body.provenance).toBe('emergency_escalation');

    process.env.OPENAI_API_KEY = originalKey;
    require('../src/services/ClinicalConversationService').clinicalService = originalService;
  });



  describe('Medical Image Intake for Clinician Review', () => {
    it('accepts valid JPEG base64 image and records for clinician review', async () => {
      const dummyBase64 = Buffer.from('fake-image-bytes').toString('base64');
      const res = await request(app)
        .post('/api/chat/upload-image')
        .send({
          imageBase64: dummyBase64,
          filename: 'skin_rash.jpg',
          mimeType: 'image/jpeg',
        });

      expect(res.status).toBe(200);
      expect(res.body.status).toBe('received_for_review');
      expect(res.body.reviewMode).toBe('demo_clinician_intake');
      expect(res.body.label).toContain('clinician review / demo analysis');
    });

    it('rejects unsupported file MIME types (e.g. executable/script)', async () => {
      const res = await request(app)
        .post('/api/chat/upload-image')
        .send({
          imageBase64: 'Y29uc29sZS5sb2coImhpIik7',
          filename: 'payload.js',
          mimeType: 'application/javascript',
        });

      expect(res.status).toBe(400);
      expect(res.body.message).toContain('Unsupported MIME type');
    });
  });
});
