import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';

export default class NVIDIAProvider extends BaseProvider {
  name = 'NVIDIA';
  getApiKeyLink = 'https://build.nvidia.com/';

  config = {
    apiTokenKey: 'NVIDIA_API_KEY',
    baseUrlKey: 'NVIDIA_BASE_URL',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
  };

  staticModels: ModelInfo[] = [
    /*
     * Essential fallback models - only the most stable/reliable ones
     * GPT-4o: 128k context, 4k standard output (64k with long output mode)
     */
    { name: 'gpt-4o', label: 'GPT-4o', provider: 'NVIDIA', maxTokenAllowed: 128000, maxCompletionTokens: 4096 },

    // GPT-4o Mini: 128k context, cost-effective alternative
    {
      name: 'gpt-4o-mini',
      label: 'GPT-4o Mini',
      provider: 'NVIDIA',
      maxTokenAllowed: 128000,
      maxCompletionTokens: 4096,
    },

    // GPT-3.5-turbo: 16k context, fast and cost-effective
    {
      name: 'gpt-3.5-turbo',
      label: 'GPT-3.5 Turbo',
      provider: 'NVIDIA',
      maxTokenAllowed: 16000,
      maxCompletionTokens: 4096,
    },

    // o1-preview: 128k context, 32k output limit (reasoning model)
    {
      name: 'o1-preview',
      label: 'o1-preview',
      provider: 'NVIDIA',
      maxTokenAllowed: 128000,
      maxCompletionTokens: 32000,
    },

    // o1-mini: 128k context, 65k output limit (reasoning model)
    { name: 'o1-mini', label: 'o1-mini', provider: 'NVIDIA', maxTokenAllowed: 128000, maxCompletionTokens: 65000 },

    // Qwen Coder: 32k context, coding model
    {
      name: 'qwen/qwen3-coder-480b-a35b-instruct',
      label: 'Qwen3 Coder 480B',
      provider: 'NVIDIA',
      maxTokenAllowed: 32000,
      maxCompletionTokens: 8192,
    },

    // Kimi: 32k context, reasoning model
    {
      name: 'moonshotai/kimi-k2-instruct-0905',
      label: 'Kimi K2 Instruct',
      provider: 'NVIDIA',
      maxTokenAllowed: 32000,
      maxCompletionTokens: 8192,
    },
  ];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv?: Record<string, string>,
  ): Promise<ModelInfo[]> {
    const { apiKey, baseUrl } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'NVIDIA_BASE_URL',
      defaultApiTokenKey: 'NVIDIA_API_KEY',
    });

    if (!apiKey) {
      throw `Missing API Key configuration for NVIDIA API`;
    }

    // Use NVIDIA's models endpoint or fallback to OpenAI
    const modelsUrl = baseUrl ? `${baseUrl}/models` : 'https://integrate.api.nvidia.com/v1/models';
    const response = await fetch(modelsUrl, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
    });

    const res = (await response.json()) as any;
    const staticModelIds = this.staticModels.map((m) => m.name);

    const data = res.data.filter(
      (model: any) =>
        model.object === 'model' &&
        (model.id.startsWith('gpt-') || model.id.startsWith('o') || model.id.startsWith('chatgpt-')) &&
        !staticModelIds.includes(model.id),
    );

    return data.map((m: any) => {
      // Get accurate context window from OpenAI API
      let contextWindow = 32000; // default fallback

      // OpenAI provides context_length in their API response
      if (m.context_length) {
        contextWindow = m.context_length;
      } else if (m.id?.includes('gpt-4o')) {
        contextWindow = 128000; // GPT-4o has 128k context
      } else if (m.id?.includes('gpt-4-turbo') || m.id?.includes('gpt-4-1106')) {
        contextWindow = 128000; // GPT-4 Turbo has 128k context
      } else if (m.id?.includes('gpt-4')) {
        contextWindow = 8192; // Standard GPT-4 has 8k context
      } else if (m.id?.includes('gpt-3.5-turbo')) {
        contextWindow = 16385; // GPT-3.5-turbo has 16k context
      }

      // Determine completion token limits based on model type (accurate 2025 limits)
      let maxCompletionTokens = 4096; // default for most models

      if (m.id?.startsWith('o1-preview')) {
        maxCompletionTokens = 32000; // o1-preview: 32K output limit
      } else if (m.id?.startsWith('o1-mini')) {
        maxCompletionTokens = 65000; // o1-mini: 65K output limit
      } else if (m.id?.startsWith('o1')) {
        maxCompletionTokens = 32000; // Other o1 models: 32K limit
      } else if (m.id?.includes('o3') || m.id?.includes('o4')) {
        maxCompletionTokens = 100000; // o3/o4 models: 100K output limit
      } else if (m.id?.includes('gpt-4o')) {
        maxCompletionTokens = 4096; // GPT-4o standard: 4K (64K with long output mode)
      } else if (m.id?.includes('gpt-4')) {
        maxCompletionTokens = 8192; // Standard GPT-4: 8K output limit
      } else if (m.id?.includes('gpt-3.5-turbo')) {
        maxCompletionTokens = 4096; // GPT-3.5-turbo: 4K output limit
      }

      return {
        name: m.id,
        label: `${m.id} (${Math.floor(contextWindow / 1000)}k context)`,
        provider: 'NVIDIA',
        maxTokenAllowed: Math.min(contextWindow, 128000), // Cap at 128k for safety
        maxCompletionTokens,
      };
    });
  }

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { baseUrl } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'NVIDIA_BASE_URL',
      defaultApiTokenKey: 'NVIDIA_API_KEY',
    });
    let { apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'NVIDIA_BASE_URL',
      defaultApiTokenKey: 'NVIDIA_API_KEY',
    });

    // Use hardcoded API key if none is provided (NOT recommended for production)
    if (!apiKey) {
      apiKey = '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';
    }

    // For NVIDIA API, we need to use the authorization header format
    const openai = createOpenAI({
      baseURL: baseUrl || 'https://integrate.api.nvidia.com/v1',
      apiKey,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
    });

    return openai(model);
  }
}
