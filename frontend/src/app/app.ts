import { Component, signal, computed, inject } from '@angular/core';
import {
  PredictionService,
  PredictResponse,
  BatchPredictResponse,
} from './prediction.service';

type Tab = 'single' | 'batch';

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private readonly svc = inject(PredictionService);

  activeTab = signal<Tab>('single');

  singleText = signal('');
  singleResult = signal<PredictResponse | null>(null);
  singleLoading = signal(false);
  singleError = signal<string | null>(null);

  batchTweets = signal<string[]>(['']);
  batchResult = signal<BatchPredictResponse | null>(null);
  batchLoading = signal(false);
  batchError = signal<string | null>(null);
  pasteText = signal('');
  showPaste = signal(false);

  singleCharCount = computed(() => this.singleText().length);
  validBatchTweets = computed(() =>
    this.batchTweets().filter(t => t.trim().length > 0)
  );
  batchValidCount = computed(() => this.validBatchTweets().length);

  setTab(tab: Tab): void {
    this.activeTab.set(tab);
  }

  analyzeSingle(): void {
    const text = this.singleText().trim();
    if (!text || this.singleLoading()) return;
    this.singleLoading.set(true);
    this.singleResult.set(null);
    this.singleError.set(null);
    this.svc.predict(text).subscribe({
      next: r => {
        this.singleResult.set(r);
        this.singleLoading.set(false);
      },
      error: () => {
        this.singleError.set('Cannot reach API — is it running on localhost:8000?');
        this.singleLoading.set(false);
      },
    });
  }

  addTweet(): void {
    if (this.batchTweets().length >= 50) return;
    this.batchTweets.update(t => [...t, '']);
  }

  removeTweet(i: number): void {
    const updated = this.batchTweets().filter((_, idx) => idx !== i);
    this.batchTweets.set(updated.length ? updated : ['']);
  }

  updateTweet(i: number, value: string): void {
    this.batchTweets.update(tweets => {
      const copy = [...tweets];
      copy[i] = value;
      return copy;
    });
  }

  parsePaste(): void {
    const lines = this.pasteText()
      .split('\n')
      .map(l => l.trim())
      .filter(l => l.length > 0)
      .slice(0, 50);
    if (lines.length) {
      this.batchTweets.set(lines);
      this.pasteText.set('');
      this.showPaste.set(false);
    }
  }

  analyzeBatch(): void {
    const tweets = this.validBatchTweets();
    if (!tweets.length || this.batchLoading()) return;
    this.batchLoading.set(true);
    this.batchResult.set(null);
    this.batchError.set(null);
    this.svc.predictBatch(tweets).subscribe({
      next: r => {
        this.batchResult.set(r);
        this.batchLoading.set(false);
      },
      error: () => {
        this.batchError.set('Cannot reach API — is it running on localhost:8000?');
        this.batchLoading.set(false);
      },
    });
  }

  confidence(r: PredictResponse): number {
    return r.prediction === 1 ? r.probability_disaster : r.probability_not_disaster;
  }

  confPct(r: PredictResponse): string {
    return (this.confidence(r) * 100).toFixed(1);
  }

  confBarWidth(r: PredictResponse): number {
    return this.confidence(r) * 100;
  }

  barStyle(r: PredictResponse): Record<string, string> {
    return { '--bar-w': this.confBarWidth(r) + '%' } as Record<string, string>;
  }

  disasterCount(batch: BatchPredictResponse): number {
    return batch.results.filter(r => r.prediction === 1).length;
  }

  truncatedTweet(i: number): string {
    const tweet = this.validBatchTweets()[i] ?? '';
    return tweet.length > 65 ? tweet.slice(0, 65) + '…' : tweet;
  }
}
