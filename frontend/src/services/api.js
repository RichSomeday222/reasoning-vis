const API_BASE_URL = 'https://beamsearch-demo.ngrok.io';

export class BeamSearchAPI {
  async generateBeamSearch(question, beamWidth = 3) {
    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true' 
        },
        body: JSON.stringify({
          question: question,
          beam_width: beamWidth
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API call failed:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        headers: {
          'ngrok-skip-browser-warning': 'true'  // 添加这个头
        }
      });
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'error' };
    }
  }
}

export const beamSearchAPI = new BeamSearchAPI();