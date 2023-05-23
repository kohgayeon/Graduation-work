clear,clc,close all;    % 변수 지우기/명령 창 clear/figure 창 닫기
cd('D:\Dataset\s1');    % directory 변경
load sync_uwb_sig.mat   % 전처리가 된 파일 data
load sync_ref_sig.mat
load dataSum.mat

uwb_fs = 20;

uwb_data=dataSum.idx(1,:);
uwb_unionData=zeros(1,1);
len=length(uwb_data);
for i=1:len
   index=dataSum.idx(1,i);
   uwb_data=sync_uwb_sig.data{3,1}(index, i);
   uwb_unionData(i,1)=uwb_data;

end
save s1_unionData.mat s1_unionData

uwb_data=sync_uwb_sig.data{3,1}(178,:);
uwb_temp = [1:length(uwb_data)]; % [1:uwb_data filtering data의 길이] 벡터 생성
time_uwb = uwb_temp/uwb_fs;

union_temp=[1:length(uwb_unionData)];
time_union=union_temp/uwb_fs;

figure; % 디폴트 Figure를 생성
subplot(211);
plot(time_union, uwb_unionData); axis tight; title('uwb signal, union data'); xlabel('Time (s)');
subplot(212);
plot(time_uwb, uwb_data); axis tight; title('uwb signal, 178 index'); xlabel('Time (s)');