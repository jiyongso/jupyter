function RMC_imagesc(Source_r,Source_theta,Source_phi,reshape_lamda)
figure;
hold on;
imagesc(0:90,0:350,reshape_lamda);
plot(Source_theta,Source_phi,'ro','MarkerSize',45,'MarkerEdgeColor','k','LineWidth',2)

title(['Source position at (', num2str(Source_r),',' num2str(Source_theta),',' num2str(Source_phi),')'],'fontsize',25,'fontweight','bold')
xlim([0,90])
ylim([0,350])
xlabel('Theta (deg)','fontsize',20,'fontweight', 'bold')
ylabel('Phi (deg)','fontsize',20,'fontweight', 'bold')
axis xy
colormap(flipud(gray.^3))
colorbar;
%title(titlename);
end