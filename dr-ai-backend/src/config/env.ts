import dotenv from 'dotenv';
dotenv.config();

const INSECURE_DEFAULTS = [
  'your-secret-key',
  'dr-ai-jwt-secret-key-production-secure',
  'dr-ai-super-secure-production-secret-key-256',
  'secret',
  'jwtsecret',
  'change-this-to-a-secure-random-secret-key-in-production',
];

export interface AppConfig {
  PORT: number;
  NODE_ENV: 'development' | 'production' | 'test';
  MONGODB_URI: string;
  JWT_SECRET: string;
  OPENAI_API_KEY?: string;
  OPENAI_MODEL: string;
  CORS_ORIGIN: string;
  DEMO_MODE: boolean;
}

export function validateConfig(): AppConfig {
  const NODE_ENV = (process.env.NODE_ENV as 'development' | 'production' | 'test') || 'development';
  const PORT = parseInt(process.env.PORT || '5000', 10);
  if (isNaN(PORT) || PORT <= 0 || PORT > 65535) {
    throw new Error('[FATAL] Startup halted: Invalid PORT configured. Must be between 1 and 65535.');
  }
  const JWT_SECRET = process.env.JWT_SECRET || (NODE_ENV === 'test' ? 'test-environment-jwt-secret-key-at-least-32-chars!' : '');
  const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/dr-ai';
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY || undefined;
  const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o';
  const CORS_ORIGIN = process.env.CORS_ORIGIN || '*';
  const DEMO_MODE = process.env.DEMO_MODE === 'true' || process.env.DEMO_MODE === undefined;

  if (NODE_ENV === 'production') {
    if (!JWT_SECRET) {
      throw new Error('[FATAL] Production startup halted: JWT_SECRET environment variable is required.');
    }
    if (INSECURE_DEFAULTS.includes(JWT_SECRET.toLowerCase())) {
      throw new Error('[FATAL] Production startup halted: JWT_SECRET must not use a known insecure placeholder.');
    }
    if (JWT_SECRET.length < 32) {
      throw new Error('[FATAL] Production startup halted: JWT_SECRET must be at least 32 characters long.');
    }
  }

  // Fallback for dev only with clear warning
  const effectiveJwtSecret = JWT_SECRET || 'dev-only-secret-key-must-be-replaced-in-prod-32chars!';
  if (!JWT_SECRET && NODE_ENV !== 'test') {
    console.warn('⚠️ [SECURITY WARNING] JWT_SECRET is not set in environment. Using ephemeral development secret. Do NOT use in production.');
  }

  return {
    PORT,
    NODE_ENV,
    MONGODB_URI,
    JWT_SECRET: effectiveJwtSecret,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    CORS_ORIGIN,
    DEMO_MODE,
  };
}

export const config = validateConfig();
