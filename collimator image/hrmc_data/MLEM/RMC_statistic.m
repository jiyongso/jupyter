function [SNR,PSNR,SSIM]=RMC_statistic(result_image,real_image)
MSE=abs(sum(sum((real_image-result_image).^2)))./360;
SNR=10*log10(sum(sum(result_image.^2))/(MSE));
PSNR=10*log10((max(result_image).^2)/(MSE));
SSIM=ssim(result_image,real_image);
%PSNR=10*log10((alpha/2)^2/(MSE));