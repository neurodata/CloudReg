%     % local rotation
%     theta = 50;
%     
%     bx = -3000;
%     by = -1000;
%     
% %    cx = -2000;
%     cx = 500;
%     cy = 3500;
%     
%     rotmat = [cos(theta),-sin(theta);
%         sin(theta),cos(theta)];
%     ROTX = cos(theta*pi/180)*(XI-cx) + sin(theta*pi/180)*(YI-cy) - (XI-cx);
%     ROTY = -sin(theta*pi/180)*(XI-cx) + cos(theta*pi/180)*(YI-cy) - (YI-cy);
%     blob_width = 3000;
%     blob = exp(-((XI - bx).^2 + (YI-by).^2 + (ZI.^2))/2/(blob_width)^2);
%     %x_idx = (XI - bx) > 0;
%     % testing uniform rotation
%     %blob = zeros(size(XI));
%     %blob(x_idx) = 0.5;
%     for t = 1 : nT
%         vty(:,:,:,t) = ROTY.*blob;
%         vtx(:,:,:,t) = ROTX.*blob;
%     end

%    % initial local translation
%   blob_width = 3000;
%   blob_displacement = 3000;
%   bx2 = -5000;
%   by2 = 0;
%   initial_y_disp = exp(-((XI - bx2).^2 + (YI - by2).^2 + (ZI).^2)/2/(blob_width)^2) * blob_displacement;
%   for t = 1 : nT
%       vty(:,:,:,t) = vty(:,:,:,t) + initial_y_disp;
%   end