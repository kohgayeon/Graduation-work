%cd('D:\Dataset\s2'); % directory 변경
%load uwb_sliced.mat
load CutDatas1.mat
%load sync_ref_sig.mat

%uwb_sig=signal(:,:);
%ref_stage=stage(:,:);

%uwb_sig=sync_uwb_sig.data{3,1}(178,:);
uwb_sig=cut_data;
%ref_stage=sync_ref_sig.stg;
ref_stage=cut_stg;
%%

uwb_fs = 20;

[b1,a1] = butter(5,0.1/(uwb_fs/2), 'high'); % 0.1 Hz cutoff frequency - High pass filtering
[b2,a2] = butter(5,4/(uwb_fs/2), 'low'); % 4 Hz cutoff frequency - low pass filtering

f_sig = filtfilt(b1,a1, uwb_sig); % high pass filtering 적용
ff_sig = filtfilt(b2,a2, f_sig); % low pass filtering 적용

% Filtering check
st_time = 12001;
end_time = st_time+600-1;
figure;
subplot(211); plot(uwb_sig(st_time : end_time)); axis tight; 
subplot(212); plot(ff_sig(st_time : end_time)); axis tight;

tms = (0:numel(ff_sig)-1)/uwb_fs;
[cfs, frq] = cwt(ff_sig, uwb_fs, 'FrequencyLimits', [0 5]);

% CWT check
figure,
%subplot(211), plot(ff_sig(st_time : end_time)); axis tight;
%subplot(212), 
surface(tms(st_time : end_time), frq, abs(cfs(:, st_time : end_time))); axis tight; shading flat; 

CWTData.fs = uwb_fs;
CWTData.filtData = ff_sig;
CWTData.time = tms;
CWTData.freq = frq;
CWTData.Power = abs(cfs);
CWTData.stg = sync_ref_sig.stg;

save CWTData_s2.mat CWTData
