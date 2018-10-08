%download nsltools at https://isr.umd.edu/Labs/NSL/Software.htm and add to path
rv=[1, 2, 4, 8, 16, 32];
sv=[0.5, 1, 2, 4, 8];
loadload;
%aud_orig = original auditory spectrogram
cor_orig = aud2cor(aud_orig',[10 10 -2 0 0 0 1],rv,sv,'tmp');
T=squeeze(mean(abs(cor_orig),3));
%aud_pred = predicted auditory spectrogram
%auditory spectrogram can be calculated using aud = wav2aud(waveform,[10 10 -2 0]); at 16kHz
cor_pred = aud2cor(aud_pred,[10 10 -2 0 0 0 1],rv,sv,'tmp');
N=squeeze(mean(abs(cor_pred),3));
tmp=(T-N).^2;
STMI=1-sum(tmp(:))/sum(T(:).^2);