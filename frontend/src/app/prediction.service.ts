import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictResponse {
  prediction: number;
  label: string;
  probability_disaster: number;
  probability_not_disaster: number;
  model_version: string;
  inference_time_ms: number;
}

export interface BatchPredictResponse {
  results: PredictResponse[];
  total: number;
  total_time_ms: number;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = 'http://localhost:8000';

  predict(text: string): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(`${this.baseUrl}/predict`, { text });
  }

  predictBatch(texts: string[]): Observable<BatchPredictResponse> {
    return this.http.post<BatchPredictResponse>(`${this.baseUrl}/predict/batch`, {
      tweets: texts.map(text => ({ text }))
    });
  }
}
