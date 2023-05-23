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