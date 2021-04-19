clc;
clear all;

%% yn 생성
Source_r=25;                         % input에 따라서 바꾸기
Source_theta=30;                     % input에 따라서 바꾸기 
Source_phi=0;                        % input에 따라서 바꾸기  [
  
iteration_num=1000;
activity=1;
load('C:\Users\KHS\Desktop\MLEM\system matrix_sym_50cm_czt.mat')


%% interpolation 
y=xlsread(['(' num2str(Source_r) ',' num2str(Source_theta) ',' num2str(Source_phi) ')_MCNP.xlsx']);
max_val=max(y);
yn=y./max_val;

point=length(y);

x=zeros(point,1);
for i=1:point
    x(i,1)=i*10-10;
end

% max 값에 normalization
max_val=max(y);
nor_y=y./max_val;

% interpolation 수행 
xn=0:1:max(x);
yn=interp1(x,nor_y,xn,'spline')';
%yn=interp1(x,nor_y,xn)';

%% MELM
grid_theta=10;  
grid_phi=36;   

A_GPU=sys_mat; % system matrix

% max 값에 normalization
Max_val=zeros(1,grid_phi*grid_theta);
for tttt=1:grid_phi*grid_theta
    Max_val(1,tttt)=max(A_GPU(:,tttt));
end

D_GPU=zeros(361,grid_theta*grid_phi);
for ii=1:361
   for jj=1:(grid_theta*grid_phi)
       D_GPU(ii,jj)=A_GPU(ii,jj)/Max_val(1,jj);
   end
end


A_GPU=D_GPU;
lamda_ori=ones(grid_theta*grid_phi,1);
I=ones(361,1);
a_GPU=A_GPU'*I;
iii3=0;

tic
%% 실제 선원의 위치에 해당하는 픽셀에 선원 위치시키기 

real_lamda=zeros(360,1);
% if Source_theta>0
    real_lamda(grid_phi/10*Source_theta+(Source_phi/10+1))=activity;
% else
%     real_lamda(180+Source_phi+(-1*Source_theta/10+1))=activity;
% end
%imshow(reshape(real_lamda,[grid_theta grid_phi]));

SNR=zeros(iteration_num,1);
PSNR=zeros(iteration_num,1);
SSIM=zeros(iteration_num,1);
output_image=zeros(grid_phi,grid_theta,1000);

%% MLEM (original)
tic
while(iii3<iteration_num) 
    A_lamda_ori=A_GPU*lamda_ori;
    lamda_ori=lamda_ori.*(A_GPU'*(yn./(A_lamda_ori)))./(a_GPU);
    
    output_ori=1.*lamda_ori;  
    
    iii3=iii3+1;
    [SNR(iii3), PSNR(iii3),SSIM(iii3)]=RMC_statistic(output_ori,real_lamda); 
    output_image(:,:,iii3)=reshape(lamda_ori,[grid_phi,grid_theta]);
end

RMC_imagesc(Source_r,Source_theta,Source_phi,output_image(:,:,end))
figure;
plot(1:iii3,SNR,'-xr');
title('Iteration-SNR(original)')
xlabel('Iteration number')
ylabel('SNR')
figure
plot(1:iii3,PSNR,'-ob');
title('Iteration-PSNR(original)')
xlabel('Iteration number')
ylabel('PSNR')
figure
plot(1:iii3,SSIM,'-ob');
title('Iteration-SSIM(original)')
xlabel('Iteration number')
ylabel('SSIM')


%% MLEM (Regularizing)
SNR_regul=zeros(iteration_num,1);
PSNR_regul=zeros(iteration_num,1);
SSIM_regul=zeros(iteration_num,1);
output_image_regul=zeros(grid_phi,grid_theta,1000);

r=2;
C=0.25;
lamda=ones(grid_theta*grid_phi,1);
iii4=0;

while(iii4<iteration_num) 
    A_lamda=A_GPU*lamda;
    %lamda_ori=lamda_ori.*(A_GPU'*(yn./(A_lamda)))./(a_GPU);
    
    lamda=lamda.*(A_GPU'*(yn./(A_lamda)))./(a_GPU+C*lamda_ori.^(-r));
    output=1.*lamda;  
    
    iii4=iii4+1;
    [SNR_regul(iii4), PSNR_regul(iii4),SSIM_regul(iii4)]=RMC_statistic(output,real_lamda); 
    output_image_regul(:,:,iii4)=reshape(lamda,[grid_phi,grid_theta]);
end

time=toc;

RMC_imagesc(Source_r,Source_theta,Source_phi,output_image_regul(:,:,end))
figure;
plot(1:iii4,SNR_regul,'-xr');
title('Iteration-SNR')
xlabel('Iteration number')
ylabel('SNR')
figure
plot(1:iii4,PSNR_regul,'-ob');
title('Iteration-PSNR')
xlabel('Iteration number')
ylabel('PSNR')
figure
plot(1:iii4,SSIM_regul,'-ob');
title('Iteration-SSIM')
xlabel('Iteration number')
ylabel('SSIM')