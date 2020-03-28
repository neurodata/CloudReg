clear all;
close all;
fclose all;


addpath ./Functions/
addpath ./Functions/plotting/
addpath ./Functions/nrrd/
addpath ./Functions/avwQuiet/
addpath ./Functions/downsample/
addpath ./Functions/spatially_varying_polynomial/
addpath ./Functions/textprogressbar/

%%
% note that this code should be run twice
% the first time at 100 micron
% then the second time at 50 micron
% outputs for Vikram
% we need to transform data to match the 10 micron atlas
%template_name_10 = '/data/vikram/registration_daniel_matlab/average_template_10.nrrd/';
%label_name_10 = '/data/vikram/registration_daniel_matlab/annotation_10.nrrd';

%indir = '/data/vikram/registration_daniel_matlab/Gad2_VGat_Brain12_20190308_downsampled/';

fixed_scale = 1.3; % I used 1.15, works with gauss newton uniform scale, turns off when set to 0
missing_data_correction = 0;

downloop_start = 1
for downloop = downloop_start : 2
    p = '/home/ubuntu/'
    % input this output prefix
    prefix = [p 'gad2_812_GN_registration_weights_danupdate/'];
    target_name = [p 'Gad2_812_ch1.tif'];

    in_prefix = [p '/MBAC/registration/atlases/'];
    
    % pixel size is required here as the tif data structure does not store it
    dxJ0 = [9.36 9.36  5];
%    dxJ0 = [5.0 37.44 37.44];

%    if downloop == 1
%        template_name = strcat(in_prefix,'/average_template_200.nrrd');
%        label_name = strcat(in_prefix, '/annotation_200.nrrd');
%        
%    elseif downloop == 2
    if downloop == 1
        template_name = strcat(in_prefix,'/average_template_100.nrrd');
        label_name = strcat(in_prefix, '/annotation_100.nrrd');

    elseif downloop == 2
        template_name = strcat(in_prefix,'/average_template_50.nrrd');
        label_name = strcat(in_prefix, '/annotation_50.nrrd');
        
%    elseif downloop == 3
%        template_name = strcat(in_prefix,'/average_template_50.nrrd');
%        label_name = strcat(in_prefix, '/annotation_50.nrrd');
    end
    
    
    
    % process some input strings for compatibility with downloop
    if downloop == 1
        vname = ''; % input mat file to restore v, empty string if not restoring
        Aname = ''; % input mat file to restore A, empty string if not restoring
    else
        [a,b,c] = fileparts(prefix);
        vname = [a,filesep, b,['downloop_' num2str(downloop-1) '_'], c , 'v.mat'];
        Aname = [a,filesep, b,['downloop_' num2str(downloop-1) '_'], c , 'A.mat'];

    end
    % add down loop to prefix
    [a,b,c] = fileparts(prefix);
    prefix = [a,filesep, b,['downloop_' num2str(downloop) '_'], c];
    
    %%
    [a,b,c] = fileparts(prefix);
    if ~exist(a,'dir')
        mkdir(a);
    end
    
    
    %%
    % allen atlas
    [I,meta] = nrrdread(template_name);
%    I = niftiread(template_name);
%    meta = niftiinfo(template_name);
    I = double(I);
    %  convert pixel dimensions from  mm to um
%    dxI = meta.PixelDimensions * 1000.0
    dxI = diag(sscanf(meta.spacedirections,'(%d,%d,%d) (%d,%d,%d) (%d,%d,%d)',[3,3]))';
    
    
    
    % want padding of 1mm
    npad = round(1000/dxI(1));
    I = padarray(I,[1,1,1]*npad,0,'both');
    
    
    % scale it for numerical stability, since its scale doesn't matter
    I = I - mean(I(:));
    I = I/std(I(:));
    nxI = [size(I,2),size(I,1),size(I,3)];
    xI = (0:nxI(1)-1)*dxI(1);
    yI = (0:nxI(2)-1)*dxI(2);
    zI = (0:nxI(3)-1)*dxI(3);
    xI = xI - mean(xI);
    yI = yI - mean(yI);
    zI = zI - mean(zI);
    danfigure(1);
    sliceView(xI,yI,zI,I);
    saveas(gcf,[prefix 'example_atlas.png'])
    
    [XI,YI,ZI] = meshgrid(xI,yI,zI);
    fxI = (0:nxI(1)-1)/nxI(1)/dxI(1);
    fyI = (0:nxI(2)-1)/nxI(2)/dxI(2);
    fzI = (0:nxI(3)-1)/nxI(3)/dxI(3);
    [FXI,FYI,FZI] = meshgrid(fxI,fyI,fzI);
    
    [L, meta] = nrrdread(label_name);
%    L = niftiread(label_name);
%    meta_L = niftiinfo(label_name)
    %  convert pixel dimensions from  mm to um
%    dxL = meta.PixelDimensions * 1000.0
    dxL = diag(sscanf(meta.spacedirections,'(%d,%d,%d) (%d,%d,%d) (%d,%d,%d)',[3,3]))';
    L = padarray(L,[1,1,1]*npad,0,'both');
    
    
    %%
    % vikram mouse
    info = imfinfo(target_name);
    %%
    % downsample to about same res as atlas
    down = round(dxI./dxJ0);
    textprogressbar('reading target: ');
    num_slices = length(info)
    for f = 1 : num_slices
        textprogressbar((f/num_slices)*100);
        %disp(['File ' num2str(f) ' of ' num2str(length(info))])
        J_ = double(imread(target_name,f));
        if f == 1
            nxJ0 = [size(J_,2),size(J_,1),length(info)];
            nxJ = floor(nxJ0./down);
            J = zeros(nxJ(2),nxJ(1),nxJ(3));
            WJ = zeros(nxJ(2),nxJ(1),nxJ(3));
        end
        % downsample J_
        Jd = zeros(nxJ(2),nxJ(1));
    	WJd = zeros(size(Jd)); % when there is no data, we have value 0
        for i = 1 : down(1)
            for j = 1 : down(2)
                Jd = Jd + J_(i:down(2):down(2)*nxJ(2), j:down(1):down(1)*nxJ(1))/down(1)/down(2);
		WJd = WJd + double((J_(i:down(2):down(2)*nxJ(2), j:down(1):down(1)*nxJ(1))/down(1)/down(2)>0));
            end
        end
        
        slice = floor( (f-1)/down(3) ) + 1;
        if slice > nxJ(3)
            break;
        end
        J(:,:,slice) = J(:,:,slice) + Jd/down(3);
    	WJ(:,:,slice) = WJ(:,:,slice) + WJd/down(3);
        
        if ~mod(f-1,10)
            danfigure(1234);
            imagesc(J(:,:,slice));
            axis image
            danfigure(1235);
            imagesc(WJ(:,:,slice));
            axis image	    
            drawnow;
        end
    end
    textprogressbar('done reading target.');
    display(size(J))
    dxJ = dxJ0.*down;
    xJ = (0:nxJ(1)-1)*dxJ(1);
    yJ = (0:nxJ(2)-1)*dxJ(2);
    zJ = (0:nxJ(3)-1)*dxJ(3);
    
    xJ = xJ - mean(xJ);
    yJ = yJ - mean(yJ);
    zJ = zJ - mean(zJ);


    % this isn't used anymore
    % because we  want to initialize high res
    %  coeffs  with low res coeffs
    %  10/2/19 -- VC
%    if downloop == 1
%        xJ_downloop1 = xJ;
%        yJ_downloop1 = yJ;
%        zJ_downloop1 = zJ;
%    end


    
    % J = avw.img;
    % J(isnan(J)) = 0;
    % danfigure(2);
    % sliceView(xJ,yJ,zJ,J)
    % [XJ,YJ,ZJ] = meshgrid(xJ,yJ,zJ);

    nplot = 5;

    J0 = J; % save it
    J0_orig = J0;
    
    danfigure(2);
    sliceView(xJ,yJ,zJ,J0_orig);
    saveas(gcf,[prefix 'example_target.png'])
    
    %%
    % missing data correction
    if missing_data_correction
        WJ = WJ/max(WJ(:));
        q = 0.01;
        c = quantile(J(WJ==1),q);
        J_ = J;
        J_ = J_.*(WJ) + c*(1-WJ);
        danfigure(22)
        sliceView(xJ,yJ,zJ,J_,nplot)
        J0_orig = J_;
    else
        WJ = 1;
    end

    %%
    % grid correction
    Jsum = sum(J0_orig,3);
    % blur
    grid_correction_blur_width = 200;
    [XJgrid,YJgrid] = meshgrid(xJ,yJ);
    
    K = exp(-(XJgrid.^2 + YJgrid.^2)/2/(grid_correction_blur_width)^2);
    K = K / sum(K(:));
    Ks = ifftshift(K);
    Kshat = fftn(Ks);
    Jb = ifftn(fftn(Jsum).*Kshat,'symmetric');
    figure;imagesc(Jb)
    Jtest = bsxfun(@times, J0_orig, Jb./(Jsum+1));
    figure;
    imagesc(xJ,yJ,Jtest(:,:,round(size(Jtest,3)/2)));
    axis image
    J0 = Jtest;
    
    
    danfigure(3);
    sliceView(xJ,yJ,zJ,J0);
    axis image
    saveas(gcf,[prefix 'example_target_grid.png'])
    
    
    %%
    % basic inhomogeneity correction based on histogaam flow
    % first find a low threshold for taking logs
    J = J0;
    
    range = [min(J(:)), max(J(:))];
    range = mean(range) + [-1,1]*diff(range)/2*1.25;
    
    nb = 200; % pretty good
    nb = 300; % better
    
    bins = linspace(range(1),range(2),nb);
    db = (bins(2)-bins(1));
    width = db*2;
    hist_ = zeros(1,nb);
    for b = 1 : nb
        hist_(b) = sum(exp(-(J(:) - bins(b)).^2/2/width^2)/sqrt(2*pi*width^2),1);
    end
    figure;
    plot(bins,hist_)
    thresh = bins(find(hist_==max(hist_),1,'first'))*0.5;
    
    
    
    J(J<thresh) = thresh;
    J = log(J);
    Jbar = mean(J(:));
    Jstd = std(J(:));
    J = J - Jbar;
    J = J/Jstd;
    
    % about 1 mm of padding
    padtemp = round(1000/dxI(1));
    
    J = padarray(J,[1,1,1]*padtemp,'symmetric');
    xJp = (0:size(J,2)-1)*dxJ(1);
    yJp = (0:size(J,1)-1)*dxJ(2);
    zJp = (0:size(J,3)-1)*dxJ(3);
    
    xJp = xJp - mean(xJp);
    yJp = yJp - mean(yJp);
    zJp = zJp - mean(zJp);
    
    
    [XJ,YJ,ZJ] = meshgrid(xJp,yJp,zJp);
    % K = exp(-(XJ.^2 + YJ.^2 + ZJ.^2)/2/(dxJ(1)*15)^2);
    % width = 750; % this value gives goood results
    % width = 500;
    width = 1000;
    K = exp(-(XJ.^2 + YJ.^2 + ZJ.^2)/2/(width)^2);
    K = K / sum(K(:));
    Ks = ifftshift(K);
    Kshat = fftn(Ks);
    
    
    
    % %%
    close all;
    danfigure(14);
    sliceView(xJ,yJ,zJ,exp(J))
    
    
    % iterate
    if missing_data_correction
        niterhom = 20;
    else
        niterhom = 10; % a little more for more inhomogeneity correction
    end
    textprogressbar('correcting inhomogeneity: ');
    for it = 1 : niterhom
        textprogressbar((it/niterhom)*100);
        range = [min(J(:)), max(J(:))];
        range = mean(range) + [-1,1]*diff(range)/2*1.25;
        
        bins = linspace(range(1),range(2),nb);
        db = (bins(2)-bins(1));
        width = db*1;
        
        
        hist_ = zeros(1,nb);
        for b = 1 : nb
            hist_(b) = sum(exp(-(J(:) - bins(b)).^2/2/width^2)/sqrt(2*pi*width^2),1);
        end
        danfigure(10);
        plot(bins,hist_)
        dhist = gradient(hist_,db);
        % now interpolate
        F = griddedInterpolant(bins,dhist,'linear','nearest');
        histgrad = reshape(F(J(:)),size(J));
        % I don't really like this although I do like the sign (tiny slope in flat
        % regions)
        histgrad = sign(histgrad);
        
        danfigure(11);
        sliceView(xJ,yJ,zJ,histgrad,5,[-1,1]);
        histgrad = ifftn(fftn(histgrad).*Kshat,'symmetric');
        danfigure(12);
        sliceView(xJ,yJ,zJ,histgrad);
        
        ep = 2e-1;
        ep = 1e-1;
        
        
        J = J + ep*histgrad;
        
        % standardize
        J = J - mean(J(:));
        J = J / std(J(:));
        
        danfigure(13);
        sliceView(xJ,yJ,zJ,exp(J))
        
        
%        disp(['Finished it ' num2str(it)])
        drawnow
        
        
        
    end
    textprogressbar('done correcting inhomogeneity');

    J = exp(J(padtemp+1:end-padtemp,padtemp+1:end-padtemp,padtemp+1:end-padtemp));
    J = J - mean(J(:));
    J = J/std(J(:));
    
    danfigure(3);
    sliceView(xJ,yJ,zJ,J);
    saveas(gcf,[prefix 'example_target_grid_hom.png'])
    
    %%
    close all;
    
    
    %%
    % set up target grid points included padded grid points for better boundary
    % conditions
    [XJ,YJ,ZJ] = meshgrid(xJ,yJ,zJ);
    
    fxJ = (0:nxJ(1)-1)/nxJ(1)/dxJ(1);
    fyJ = (0:nxJ(2)-1)/nxJ(2)/dxJ(2);
    fzJ = (0:nxJ(3)-1)/nxJ(3)/dxJ(3);
    [FXJ,FYJ,FZJ] = meshgrid(fxJ,fyJ,fzJ);
    
    xJp = [xJ(1)-dxJ(1), xJ, xJ(end)+dxJ(1)];
    yJp = [yJ(1)-dxJ(2), yJ, yJ(end)+dxJ(2)];
    zJp = [zJ(1)-dxJ(3), zJ, zJ(end)+dxJ(3)];
    
    %%
    % now we map them!
    %%
    nT = 5;
    dt = 1/nT;
    sigmaM = std(J(:));
    %sigmaA = sigmaM*10; % artifact
    CA = 1; % estimate
    % I want to make it less, its actually quite low
    sigmaB = sigmaM/2;
    CB = -1;
    %WJ = 1;
    sigmaC = 5.0;
    % try more
%    sigmaC = 10.0;
%    % vikram testing out even more
%    sigmaC = 20.0;
    
    
    
    danfigure(1);
    sliceView(xI,yI,zI,I)
    climI = get(gca,'clim');
    danfigure(2);
    sliceView(xJ,yJ,zJ,J)
    climJ = get(gca,'clim');
    % May 2019
    order = 4;
    nM = 1;
    nMaffine = 1; % number of m steps per e step durring affine only

    % number of affine only iterations
    naffine = 0;
    
    
    % total number of iterations
    niter = 5000;
    if downloop > 1
        niter = 500;
    end
%    niter_a5 = 1000;
%    niter_a4 = 500;
    %if dxI < 50
    %    niter = 500; % if high resolution do fewer iterations
    %end
    
    % % test!
    % niter = 10;
    % if dxI(1) < 100
    %     niter = 1; % if high resolution do fewer iterations
    % end
    
    
    % above sigma was too small, need more deformation
    % sigmaR = 5e3;
    
    % kernel width
    a = 500;
    % try different smoothness scale for
    % higher resolution to account for serious
    % local deformation

%    if downloop >= 2
%        a = 250
%    end
    p = 2;
    % apre = 1000;
    % make this a function of voxel size
    % to speed up optimization
    % 10/2/19 -- VC
    apre = 1000;
    ppre = 2;
    %aC = 2000; % I think this should be bigger, about 20 voxels, I think 2000 is too big
    %aC = 1000;
%    aC = 7.5*dxI(1); % try a little smaller
    aC = 750; % try a little smaller
    %aC = 750*2; % try bigger for rat, maybe 2x
    pC = 2;
    
    LL = (1 - 2 * a^2 * ( (cos(2*pi*dxI(1)*FXI) - 1)/dxI(1)^2 + (cos(2*pi*dxI(2)*FYI) - 1)/dxI(2)^2 + (cos(2*pi*dxI(3)*FZI) - 1)/dxI(3)^2 )).^(2*p);
    Khat = 1.0./LL;
    
    
    LLpre = (1 - 2 * apre^2 * ( (cos(2*pi*dxI(1)*FXI) - 1)/dxI(1)^2 + (cos(2*pi*dxI(2)*FYI) - 1)/dxI(2)^2 + (cos(2*pi*dxI(3)*FZI) - 1)/dxI(3)^2 )).^(2*ppre);
    Khatpre = 1.0./LLpre;
    
    LC = (1 - 2 * aC^2 * ( (cos(2*pi*dxJ(1)*FXJ) - 1)/dxJ(1)^2 + (cos(2*pi*dxJ(2)*FYJ) - 1)/dxJ(2)^2 + (cos(2*pi*dxJ(3)*FZJ) - 1)/dxJ(3)^2 )).^(pC);
    LLC = LC.^2;
    iLC = 1.0/LC;
    KhatC = 1.0./LLC;
    
    
    
    eT = 2e-6;
    eL = 1e-13; % okay seems fine
    post_affine_reduce = 0.1;
    
    %eV = 1e6;
    eV = 1e6;
    
    
    sigmaR = 5e3;
    % make it smaller 
%    sigmaR = sigmaR*2;

    
    % decrease sigmaA from x10 to x2
%    sigmaA = sigmaM*2;
    sigmaB = sigmaM * 2;
    sigmaA = sigmaM * 5;
    prior = [0.89,0.1,0.01];
    prior = [0.79, 0.2, 0.01];
    prior = prior / sum(prior);
	


    % try more timesteps because deformation is getting big
    nT = 10;
    
    
    
    % %%
    % initialize
    A = eye(4);
    A = [0,-1,0,0;
        1,0,0,0;
        0,0,1,0
        0,0,0,1]*A;
    A = [0,0,1,0;
        0,1,0,0;
        1,0,0,0;
        0,0,0,1]*A;
    % translation down in axis 2
%    A(3,4)=-500;
    % translation up in axis 1 direction
%    A(2,4)=-1400;
    % translation in axis 0
%    A(1,4)=-1500;
    %  30 degree  rotation
    %A = [0.8660254,-0.5,0,0;
    %     0.5,0.8660254,0,0;
    %     0,0,1,0
    %     0,0,0,1]*A;

%    %  5 degree clockwise rotation in xy
%    A = [0.9961947,0.0871557,0,0;
%         -0.0871557,0.9961947,0,0;
%         0,0,1,0
%         0,0,0,1]*A;
    %  10 degree  rotation in yz
%    A = [  1.0000000,  0.0000000,  0.0000000, 0;
%           0.0000000,  0.9848077, -0.1736482, 0;
%           0.0000000,  0.1736482,  0.9848077, 0;
%	   0.0000000,  0.0000000,  0.0000000, 1.0 ]*A;
%    %  15 degree  rotation
%    A = [0.9659258,-0.2588190,0,0;
%         0.2588190,0.9659258,0,0;
%         0,0,1,0
%         0,0,0,1]*A;
%    A = [0,0,1,0;
%        1,0,0,0;
%        0,1,0,0;
%        0,0,0,1]*A;
    % % note this has det -1!
%    A = diag([-1,1,1,1])*A;

    % expand atlas
    %A = diag([1.85,1.85,1.85,1])*A;
    
    
    % shrink atlas to make affine estimate more accurate
    % comment out below for SertCre
    %  stretch atlas in y dimension, typical deformation introduced
%    A = diag([1.05,0.90,0.95,1])*A;

%    A = diag([1,1.5,0.95,1])*A;
    
    % 30 degree rotation for gad2cre sample
    %A = [  0.8660254, -0.5,0.0,0;
    %   0.50,  0.8660254,0.0,0.0;
    %      0.0,  0.0,  1.0 ,0.0;
    %      0,0,0,1]*A;
    %
    %
    % add some translation to move the brain
    % in it's anterior direction

    % A(2,4)=-1100
    
    % % after 25 iters
    % A = [0.0017   -0.0272    0.8835 -0.0674*1e3
    %     0.9193   -0.0057    0.0029 -1.1686*1e3
    %     0.0964   -1.0415    0.0199 -0.0594*1e3;
    %     0,0,0,1];
    % naffine = 0;
    
    
    vtx = zeros([size(I),nT]);
    vty = zeros([size(I),nT]);
    vtz = zeros([size(I),nT]);
    
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
    
 
    % add translation in X,Y and Z axes
%    A = [eye(3),[0;0;300];[0,0,0,1]]*A;
    if fixed_scale
        A = diag([fixed_scale,fixed_scale,fixed_scale,1])*A;
    end
    
    
    % load data
    if ~isempty(Aname)
        variables = load(Aname);
        A = variables.A;
    end
    if ~isempty(vname)
        variables = load(vname);
        vtx0 = variables.vtx;
        vty0 = variables.vty;
        vtz0 = variables.vtz;
        % if size does not match we will have to resample
        if size(vtx0,4) ~= size(vtx,4)
            error('Restoring v with different number of timesteps is not supported')
        end
        if any(size(vtx0)~=size(vtx))
            for t = 1 : size(vtx0,4)
                disp(['Upsampling restored velocity field ' num2str(t) ' of ' num2str(size(vtx0,4))])
                vtx(:,:,:,t) = upsample(vtx0(:,:,:,t),[size(vtx,1),size(vtx,2),size(vtx,3)]);
                vty(:,:,:,t) = upsample(vty0(:,:,:,t),[size(vtx,1),size(vtx,2),size(vtx,3)]);
                vtz(:,:,:,t) = upsample(vtz0(:,:,:,t),[size(vtx,1),size(vtx,2),size(vtx,3)]);
            end
        end
        naffine = 0;
        warning('Because you are restoring a velocity field, we are setting number of affine only steps to 0')
    end
    
    
    It = zeros([size(I),nT]);
    It(:,:,:,1) = I;
    
    
    if downloop >= 1
        % actually
        % we need an initial linear transformation to compute our first weight
        Jq = quantile(J(:),[0.1 0.9]);
        Iq = quantile(I(:),[0.1,0.9]);
        coeffs = [mean(Jq)-mean(Iq)*diff(Jq)/diff(Iq); diff(Jq)/diff(Iq)];
        % if I do a higher order transform, I should just set the nonlinear
        % components to 0
        coeffs = [coeffs;zeros(order-2,1)];
        % note that it might be convenient to work with low order at beginning and
        % then increase order
        % make the coeffs a function of space
        coeffs = reshape(coeffs,1,1,1,[]) .*  ones([size(J),order]);
    else
        %XJ_dl1,YJ_dl1,ZJ_dl1 = meshgrid(xJ_downloop1,yJ_downloop1,zJ_downloop1)
%#        F1 = griddedInterpolant({yJ_downloop1,xJ_downloop1,zJ_downloop1},squeeze(coeffs(:,:,:,1)),'linear','nearest');
%#        F2 = griddedInterpolant({yJ_downloop1,xJ_downloop1,zJ_downloop1},squeeze(coeffs(:,:,:,2)),'linear','nearest');
%#        F3 = griddedInterpolant({yJ_downloop1,xJ_downloop1,zJ_downloop1},squeeze(coeffs(:,:,:,3)),'linear','nearest');
%#        F4 = griddedInterpolant({yJ_downloop1,xJ_downloop1,zJ_downloop1},squeeze(coeffs(:,:,:,4)),'linear','nearest');
%#        coeffs = cat(4,F1(),F2(),F3(),F4())
%        F = griddedInterpolant({yJ_downloop1,xJ_downloop1,zJ_downloop1},coeffs,'linear','nearest');
        coeffs_1 = upsample(coeffs(:,:,:,1),[size(J,1),size(J,2),size(J,3)]);
        coeffs_2 = upsample(coeffs(:,:,:,2),[size(J,1),size(J,2),size(J,3)]);
        coeffs_3 = upsample(coeffs(:,:,:,3),[size(J,1),size(J,2),size(J,3)]);
        coeffs_4 = upsample(coeffs(:,:,:,4),[size(J,1),size(J,2),size(J,3)]);
        coeffs = cat(4,coeffs_1,coeffs_2,coeffs_3,coeffs_4);
    end
    
    % start
    Esave = [];
    EMsave = [];
    ERsave = [];
    EAsave = [];
    EBsave = [];
    Asave = [];
    frame_errW = [];
    frame_I = [];
    frame_phiI = [];
    frame_errRGB = [];
    frame_W = [];
    frame_curve = [];
    dt = 1/nT;
    tic
    for it = 1 : niter
        
        % deform image
        phiinvx = XI;
        phiinvy = YI;
        phiinvz = ZI;
        for t = 1 : nT * (it > naffine)
            
            
            % sample image
            if t > 1
                F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
                It(:,:,:,t) = F(phiinvy,phiinvx,phiinvz);
            end
            % update diffeo, add and subtract identity for better boundary conditions
            Xs = XI - vtx(:,:,:,t)*dt;
            Ys = YI - vty(:,:,:,t)*dt;
            Zs = ZI - vtz(:,:,:,t)*dt;
            F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
            phiinvx = F(Ys,Xs,Zs) + Xs;
            F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
            phiinvy = F(Ys,Xs,Zs) + Ys;
            F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
            phiinvz = F(Ys,Xs,Zs) + Zs;
            
        end
        F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
        phiI = F(phiinvy,phiinvx,phiinvz);
        danfigure(6791);
        sliceView(xI,yI,zI,phiI)
        
        % now apply affine, go to sampling of J
        % ideally I should fix the double interpolation here
        % for now just leave it
        B = inv(A);
        Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3)*ZJ + B(1,4);
        Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3)*ZJ + B(2,4);
        Zs = B(3,1)*XJ + B(3,2)*YJ + B(3,3)*ZJ + B(3,4);
        
        % okay if I did this together I would see
        % AphiI = I(phiinv(B x))
        % first sample phiinv at Bx
        % then sample I at phiinv Bx
        F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
        phiinvBx = F(Ys,Xs,Zs) + Xs;
        F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
        phiinvBy = F(Ys,Xs,Zs) + Ys;
        F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
        phiinvBz = F(Ys,Xs,Zs) + Zs;
        F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
        AphiI = F(phiinvBy,phiinvBx,phiinvBz);
        
        
        % now apply the linear intensity transformation
        % order is 1 plus highest power
        fAphiI = zeros(size(J));
        for o = 1 : order
            fAphiI = fAphiI + coeffs(:,:,:,o).*AphiI.^(o-1);
        end
        
        
        danfigure(3);
        sliceView(xJ,yJ,zJ,fAphiI,nplot,climJ);
        
        err = fAphiI - J;
        
        danfigure(6666);
        sliceView(xJ,yJ,zJ,cat(4,J,fAphiI,J),nplot,climJ);
        
%         return % for checking affine
if downloop == 1 && fixed_scale == 0 && it == 1
    % show 5 scales
    Asave_ = A;
    [U,S,V] = svd(A(1:3,1:3));
    s = diag(S);
    ss = [0.9,1.0,1.1,1.2,1.3,1.4,1.5];
    for ssloop = 1 : length(ss)
        s = ss(ssloop);
        A(1:3,1:3) = U * diag([s,s,s]) * V';
        B = inv(A);
        Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3)*ZJ + B(1,4);
        Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3)*ZJ + B(2,4);
        Zs = B(3,1)*XJ + B(3,2)*YJ + B(3,3)*ZJ + B(3,4);
        
        % okay if I did this together I would see
        % AphiI = I(phiinv(B x))
        % first sample phiinv at Bx
        % then sample I at phiinv Bx
        F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
        phiinvBx = F(Ys,Xs,Zs) + Xs;
        F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
        phiinvBy = F(Ys,Xs,Zs) + Ys;
        F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
        phiinvBz = F(Ys,Xs,Zs) + Zs;
        F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
        AphiI = F(phiinvBy,phiinvBx,phiinvBz);
        
        
        % now apply the linear intensity transformation
        % order is 1 plus highest power
        fAphiI = zeros(size(J));
        for o = 1 : order
            fAphiI = fAphiI + coeffs(:,:,:,o).*AphiI.^(o-1);
        end
        
        
        danfigure(6666);
        sliceView(xJ,yJ,zJ,cat(4,J,fAphiI,J),nplot,climJ);
        saveas(6666,[prefix 'test_scale_' num2str(s) '.png']);
    end
    A = Asave_;
    B = inv(A);
    Xs = B(1,1)*XJ + B(1,2)*YJ + B(1,3)*ZJ + B(1,4);
    Ys = B(2,1)*XJ + B(2,2)*YJ + B(2,3)*ZJ + B(2,4);
    Zs = B(3,1)*XJ + B(3,2)*YJ + B(3,3)*ZJ + B(3,4);
    F = griddedInterpolant({yI,xI,zI},phiinvx-XI,'linear','nearest');
    phiinvBx = F(Ys,Xs,Zs) + Xs;
    F = griddedInterpolant({yI,xI,zI},phiinvy-YI,'linear','nearest');
    phiinvBy = F(Ys,Xs,Zs) + Ys;
    F = griddedInterpolant({yI,xI,zI},phiinvz-ZI,'linear','nearest');
    phiinvBz = F(Ys,Xs,Zs) + Zs;
    F = griddedInterpolant({yI,xI,zI},I,'linear','nearest');
    AphiI = F(phiinvBy,phiinvBx,phiinvBz);
    
    
    % now apply the linear intensity transformation
    % order is 1 plus highest power
    fAphiI = zeros(size(J));
    for o = 1 : order
        fAphiI = fAphiI + coeffs(:,:,:,o).*AphiI.^(o-1);
    end
    danfigure(6666);
    sliceView(xJ,yJ,zJ,cat(4,J,fAphiI,J),nplot,climJ);
end

        
        % now a weight
        doENumber = nMaffine;
        if it > naffine; doENumber = nM; end
        if ~mod(it-1,doENumber)
            WM = 1/sqrt(2*pi*(sigmaM^2)).*exp(-1.0/2.0/sigmaM^2*err.^2) * prior(1);
            WA = 1/sqrt(2*pi*sigmaA^2)*exp(-1.0/2.0/sigmaA^2*(CA - J).^2) * prior(2);
            WB = 1/sqrt(2*pi*sigmaB^2)*exp(-1.0/2.0/sigmaB^2*(CB - J).^2) * prior(3);
            
            Wsum = WM + WA + WB;
            
            % due to numerical error, there are sum places where Wsum may be 0
            wsm = max(Wsum(:));
            wsm_mult = 1e-6;
            Wsum(Wsum<wsm_mult*wsm) = wsm_mult*wsm;
            
            WM = WM./Wsum;
            WA = WA./Wsum;
            WB = WB./Wsum;
            
            % there is probably a numerically better way to do this
            % if I took logs and then subtractedthe max and then took
            % exponentials maybe
            % not worth it, the numerics are okay
            danfigure(45);
            %         sliceView(xJ,yJ,zJ,WM);
            sliceView(xJ,yJ,zJ,cat(4,WM,WA,WB),nplot);
            
        end
        errW = err.*WM.*WJ;
        
        % now we hit the error with Df, the D is with respect to intensity
        % parameters, not space
        % note f is a map from R to R in this case, so Df is just a scalar
        Df = zeros(size(errW));
        for o = 2 : order
            % note I had a mistake here where I did fAphiI! it is now fixed
            Df = Df + (o-1)*AphiI.^(o-2).*coeffs(:,:,:,o);
        end
        errWDf = errW.*Df;
        
        
        danfigure(4);
        sliceView(xJ,yJ,zJ,errW,nplot);
        
        
        % cost
        vtxhat = fft(fft(fft(vtx,[],1),[],2),[],3);
        vtyhat = fft(fft(fft(vty,[],1),[],2),[],3);
        vtzhat = fft(fft(fft(vtz,[],1),[],2),[],3);
        ER = sum(sum(sum(LL.*sum(abs(vtxhat).^2 + abs(vtyhat).^2 + abs(vtzhat).^2,4))))/2/sigmaR^2*dt*prod(dxI)/(size(I,1)*size(I,2)*size(I,3));
        EM = sum(sum(sum((fAphiI - J).^2.*WM)))*prod(dxJ)/2/sigmaM^2;
        EA = sum(sum(sum((CA - J).^2.*WA)))*prod(dxJ)/2/sigmaA^2;
        EB = sum(sum(sum((CB - J).^2.*WB)))*prod(dxJ)/2/sigmaB^2;
        % note regarding energy
        % I also need to include the other terms (related to variance only)
        EM = EM + sum(WM(:)).*log(2*pi*sigmaM^2)/2*prod(dxJ);
        EA = EA + sum(WA(:)).*log(2*pi*sigmaA^2)/2*prod(dxJ);
        EB = EB + sum(WB(:)).*log(2*pi*sigmaB^2)/2*prod(dxJ);
        E = ER + EM + EA + EB;
        fprintf(1,'Iteration %d, energy %g, reg %g, match %g, artifact %g, background %g\n',it,E,ER,EM,EA,EB);
        Esave = [Esave,E];
        ERsave = [ERsave,ER];
        EMsave = [EMsave,EM];
        EAsave = [EAsave,EA];
        EBsave = [EBsave,EB];
        
        % first let's do affine
        % at the end, we'll update affine and coeffs
        % gradient
        [AphiI_x,AphiI_y,AphiI_z] = gradient(AphiI,dxJ(1),dxJ(2),dxJ(3));
        grad = zeros(4,4);
        do_GN = 1; % do gauss newton
        rigid_only  = 0; % constrain affine to be rigid
	    uniform_scale_only = 1; % for uniform scaling
        % NOTE
        % without Gauss Newton, the affine transformation will be updated with
        % rigid transforms.  If the initial guess is nonrigid, it wli lremain
        % nonrigid
        % with GN, the affine transformation will be projected onto rigid
        % transforms, you will lose any nonrigid initialization
        if ~do_GN % do gradient descent
        [AphiI_x,AphiI_y,AphiI_z] = gradient(AphiI,dxJ(1),dxJ(2),dxJ(3));
        grad = zeros(4,4);
        for r = 1 : 3
            for c = 1 : 4
                dA = (double((1:4)'==r)) * double(((1:4)==c));
                AdAB = A * dA * B;
                AdABX = AdAB(1,1)*XJ + AdAB(1,2)*YJ + AdAB(1,3)*ZJ + AdAB(1,4);
                AdABY = AdAB(2,1)*XJ + AdAB(2,2)*YJ + AdAB(2,3)*ZJ + AdAB(2,4);
                AdABZ = AdAB(3,1)*XJ + AdAB(3,2)*YJ + AdAB(3,3)*ZJ + AdAB(3,4);
                grad(r,c) = -sum(sum(sum(errWDf.*(AphiI_x.*AdABX + AphiI_y.*AdABY + AphiI_z.*AdABZ))))*prod(dxJ)/sigmaM^2;
            end
        end
        if rigid_only
            grad(1:3,1:3) = grad(1:3,1:3) - grad(1:3,1:3)';
        end
        else % do Gauss Newton optimization
            [fAphiI_x,fAphiI_y,fAphiI_z] = gradient(fAphiI,dxJ(1),dxJ(2),dxJ(3));
            Jerr = zeros(size(J,1),size(J,2),size(J,3),12);
            count = 0;
            for r = 1 : 3
                for c = 1 : 4
                    dA = double((1:4==r))' * double((1:4==c));
                    AdAAi = A*dA;
                    Xs = AdAAi(1,1)*XJ + AdAAi(1,2)*YJ + AdAAi(1,3)*ZJ + AdAAi(1,4);
                    Ys = AdAAi(2,1)*XJ + AdAAi(2,2)*YJ + AdAAi(2,3)*ZJ + AdAAi(2,4);
                    Zs = AdAAi(3,1)*XJ + AdAAi(3,2)*YJ + AdAAi(3,3)*ZJ + AdAAi(3,4);
                    count = count + 1;
                    Jerr(:,:,:,count) = (bsxfun(@times, fAphiI_x,Xs) + bsxfun(@times, fAphiI_y,Ys) + bsxfun(@times, fAphiI_z,Zs)).*sqrt(WM);
                end
            end
%             JerrJerr = squeeze(sum(sum(sum(bsxfun(@times, permute(bsxfun(@times,Jerr,1),[1,2,3,4,5]) , permute(Jerr,[1,2,3,5,4])),3),2),1));
            % the above line is very slow
            Jerr_ = reshape(Jerr,[],count);
            JerrJerr = Jerr_' * Jerr_;
            % step
            step = JerrJerr \ squeeze(sum(sum(sum(bsxfun(@times, Jerr, err.*sqrt(WM)),3),2),1));
            step = reshape(step,4,3)';
        end % end of affine gradient loop
        
        % now pull back the error, pad it so we can easily get 0 boundary
        % conditions
        errWDfp = padarray(errWDf,[1,1,1],0);
        phi1tinvx = XI;
        phi1tinvy = YI;
        phi1tinvz = ZI;
        % define these variables for output even if only doing affine
        Aphi1tinvx = A(1,1)*phi1tinvx + A(1,2)*phi1tinvy + A(1,3)*phi1tinvz + A(1,4);
        Aphi1tinvy = A(2,1)*phi1tinvx + A(2,2)*phi1tinvy + A(2,3)*phi1tinvz + A(2,4);
        Aphi1tinvz = A(3,1)*phi1tinvx + A(3,2)*phi1tinvy + A(3,3)*phi1tinvz + A(3,4);
        for t = nT*(it>naffine) : -1 : 1
            % update diffeo (note plus)
            Xs = XI + vtx(:,:,:,t)*dt;
            Ys = YI + vty(:,:,:,t)*dt;
            Zs = ZI + vtz(:,:,:,t)*dt;
            F = griddedInterpolant({yI,xI,zI},phi1tinvx-XI,'linear','nearest');
            phi1tinvx = F(Ys,Xs,Zs) + Xs;
            F = griddedInterpolant({yI,xI,zI},phi1tinvy-YI,'linear','nearest');
            phi1tinvy = F(Ys,Xs,Zs) + Ys;
            F = griddedInterpolant({yI,xI,zI},phi1tinvz-ZI,'linear','nearest');
            phi1tinvz = F(Ys,Xs,Zs) + Zs;
            % determinant of jacobian
            [phi1tinvx_x,phi1tinvx_y,phi1tinvx_z] = gradient(phi1tinvx,dxI(1),dxI(2),dxI(3));
            [phi1tinvy_x,phi1tinvy_y,phi1tinvy_z] = gradient(phi1tinvy,dxI(1),dxI(2),dxI(3));
            [phi1tinvz_x,phi1tinvz_y,phi1tinvz_z] = gradient(phi1tinvz,dxI(1),dxI(2),dxI(3));
            detjac = phi1tinvx_x.*(phi1tinvy_y.*phi1tinvz_z - phi1tinvy_z.*phi1tinvz_y) ...
                - phi1tinvx_y.*(phi1tinvy_x.*phi1tinvz_z - phi1tinvy_z.*phi1tinvz_x) ...
                + phi1tinvx_z.*(phi1tinvy_x.*phi1tinvz_y - phi1tinvy_y.*phi1tinvz_x);
            
            Aphi1tinvx = A(1,1)*phi1tinvx + A(1,2)*phi1tinvy + A(1,3)*phi1tinvz + A(1,4);
            Aphi1tinvy = A(2,1)*phi1tinvx + A(2,2)*phi1tinvy + A(2,3)*phi1tinvz + A(2,4);
            Aphi1tinvz = A(3,1)*phi1tinvx + A(3,2)*phi1tinvy + A(3,3)*phi1tinvz + A(3,4);
            
            % pull back error with 0 padding
            F = griddedInterpolant({yJp,xJp,zJp},(-errWDfp/sigmaM^2),'linear','nearest');
            lambda = F(Aphi1tinvy,Aphi1tinvx,Aphi1tinvz).*detjac.*abs(det(A));
            
            % get the gradient of the image
            [I_x,I_y,I_z] = gradient(It(:,:,:,t),dxI(1),dxI(2),dxI(3));
            
            % set up the gradient
            gradx = I_x.*lambda;
            grady = I_y.*lambda;
            gradz = I_z.*lambda;
            
            % kernel and reg
            % we add extra smothness here as a predconditioner
            gradx = ifftn((fftn(gradx).*Khat + vtxhat(:,:,:,t)/sigmaR^2).*Khatpre,'symmetric');
            grady = ifftn((fftn(grady).*Khat + vtyhat(:,:,:,t)/sigmaR^2).*Khatpre,'symmetric');
            gradz = ifftn((fftn(gradz).*Khat + vtzhat(:,:,:,t)/sigmaR^2).*Khatpre,'symmetric');
            
            
            % now update
            %         vtx(:,:,:,t) = vtx(:,:,:,t) - gradx*eV;
            %         vty(:,:,:,t) = vty(:,:,:,t) - grady*eV;
            %         vtz(:,:,:,t) = vtz(:,:,:,t) - gradz*eV;
            
            
            % a maximum for stability, this is a maximum but is identify for
            % small argument
            gradxeV = gradx*eV;
            gradyeV = grady*eV;
            gradzeV = gradz*eV;
            norm = sqrt(gradxeV.^2 + gradyeV.^2 + gradzeV.^2);
            mymax = 1*dxJ(1); % is this an appropriate maximum?
            % I do not think there should be a dt here
            % for this data I think 1 voxel is probably way too small
            gradxeV = gradxeV./norm.*atan(norm*pi/2/mymax)*mymax/pi*2;
            gradyeV = gradyeV./norm.*atan(norm*pi/2/mymax)*mymax/pi*2;
            gradzeV = gradzeV./norm.*atan(norm*pi/2/mymax)*mymax/pi*2;
            vtx(:,:,:,t) = vtx(:,:,:,t) - gradxeV;
            vty(:,:,:,t) = vty(:,:,:,t) - gradyeV;
            vtz(:,:,:,t) = vtz(:,:,:,t) - gradzeV;
            
            
        end
        
        
        
        basis = zeros(size(J,1),size(J,2),size(J,3),1,order);
        for o = 1 : order
            basis(:,:,:,1,o) = AphiI.^(o-1);
        end
        if it == 1
            if downloop == 1
                nitercoeffs = 10;
            else
                nitercoeffs = 20;
            end
            % vikram testing fewer because maybe better to update slower in the beginning
        else
            nitercoeffs = 5;
            % vikram testing fewer because maybe better to update slower in the beginning
        end
        coeffs = squeeze(estimate_coeffs_3d(basis,J,sqrt(WJ.*WM)/sigmaM,LC/sigmaC,coeffs,nitercoeffs));
        danfigure(466446);
        sliceView(xJ,yJ,zJ,coeffs);
        
        
        % I can also update my constants
        CB = sum(WB(:).*J(:).*WJ(:))/sum(WB(:).*WJ(:));
        CA = sum(WA(:).*J(:).*WJ(:))/sum(WA(:).*WJ(:));
        
        % update A
        if ~do_GN % if gradient descent
        e = [ones(3)*eL,ones(3,1)*eT;0,0,0,0];
        if it > naffine
            % smaller step size now!
            e = e * post_affine_reduce;
        end
        A = A * expm(-e.*grad);
    %     e.*grad % I printed to check size of gradient
        else % do gauss newton
            Ai = inv(A);
            eA = 0.2;
            Ai(1:3,1:4) = Ai(1:3,1:4) - eA * step;
            A = inv(Ai);
            if rigid_only
                %
                [U,S,V] = svd(A(1:3,1:3));
                A(1:3,1:3) = U * V';
            end
	    if uniform_scale_only
                [U,S,V] = svd(A(1:3,1:3));
		s = diag(S);
		s = exp(mean(log(s))) * ones(size(s));
		if fixed_scale ~= 0
		    s = [1,1,1]*fixed_scale;
		end
                A(1:3,1:3) = U * diag(s) *  V';
		
	    end
        end
        
        danfigure(8);
        Asave = [Asave,A(:)];
        subplot(1,3,1)
        plot(Asave([1,2,3,5,6,7,9,10,11],:)')
        title('linear part')
        subplot(1,3,2)
        plot(Asave([13,14,15],:)')
        ylabel um
        title('translation part')
        legend('x','y','z','location','best')
        subplot(1,3,3)
        plot([Esave;ERsave;EMsave;EAsave;EBsave]')
        legend('tot','reg','match','artifact','background','location','best')
        title('Energy')
        saveas(8,[prefix 'energy.png'])
        
        % let's also plot the intensity transformation
        coeffs_ = squeeze(mean(mean(mean(coeffs,1),2),3));
        danfigure(78987);
        t = linspace(climI(1),climI(2),1000)';
        out = t * 0;
        for o = 1 : order
            out = out + coeffs_(o)*t.^(o-1);
        end
        plot(t,out,'linewidth',2)
        set(gca,'xlim',[t(1),t(end)])
        set(gca,'ylim',[min(out),max(out)])
        
        set(gca,'linewidth',2)
        axis square
        set(gca,'fontsize',12)
        xlabel 'Atlas Intensity'
        ylabel 'Target Intensity'
        set(gcf,'paperpositionmode','auto')
        
        
        
        drawnow;
        
        if it <= 100 || ~mod(it-1,11)
            frame_I = [frame_I,getframe(3)];
            frame_errRGB = [frame_errRGB,getframe(6666)];
            frame_W = [frame_W,getframe(45)];
            frame_errW = [frame_errW,getframe(4)];
            frame_curve = [frame_curve,getframe(78987)];
            frame_phiI = [frame_phiI,getframe(6791)];
        end
        
        if it <= 50 || ~mod(it-1,11)
            
            frame2Gif(frame_I, [prefix 'fAphiI.gif']);
            frame2Gif(frame_phiI, [prefix 'phiI.gif']);
            frame2Gif(frame_errRGB,[prefix 'errRGB.gif']);
            frame2Gif(frame_errW,[prefix 'errW.gif'])
            frame2Gif(frame_W,[prefix 'W.gif'])
            frame2Gif(frame_curve,[prefix 'curve.gif'])
            frame2Gif(frame_errRGB(1:10:end),[prefix 'errRGBdown.gif']);
            if it == 1
                % write J once
                saveas(2,[prefix 'J.png'])
                % write I once
                saveas(1,[prefix 'I.png'])
            end
            % write deformed I every time
            saveas(3,[prefix 'fAphiI.png'])
        end
        
        
        save([prefix 'A.mat'],'A')
        if ~mod(it-1,100)
            save([prefix 'v.mat'],'vtx','vty','vtz','-v7.3')
        end
        
        
    end
    save([prefix 'v.mat'],'vtx','vty','vtz','-v7.3')
    toc
    
    
    % % apply phi to the atlas (by composing with phiinv)
    % % then apply affine transformation A to the atlas (by composing with inv(A))
    % % the atlas voxels are at the points XI,YI,ZI
    % save('phiinv.mat','phiinvx','phiinvy','phiinvz');
    % % phi1tinv
    % save('phi1tinv.mat','phi1tinvx','phi1tinvy','phi1tinvz');
    % % images
    % save('I.mat','I','XI','YI','ZI')
    % save('J.mat','J','XJ','YJ','ZJ')
    
    
    %%
    % pull back the target
    
    F = griddedInterpolant({yJp,xJp,zJp},padarray(J,[1,1,1]),'linear','nearest');
    Ji = F(Aphi1tinvy,Aphi1tinvx,Aphi1tinvz);
    figure;sliceView(Ji)
    
    % overlay the labels
    rng(1);
    colors = randn(256,3);
    colors(1,:) = 0;
    
    Lm = mod(double(L),256)+1;
    LRGB = reshape([colors(Lm(:),1),colors(Lm(:),2),colors(Lm(:),3)],[size(L),3]);
    figure;sliceView(LRGB)
    
    % opacity of labels
    alpha = 0.125;
    % scale Ji
    Js = (Ji- climJ(1))/diff(climJ);
    Js(Js>1) = 1;
    Js(Js<0) = 0;
    RGB = bsxfun(@plus, LRGB*alpha, Js*(1-alpha));
    close all;
    hf = danfigure();
    set(hf,'paperpositionmode','auto')
    nslices = 8;
    sliceView(xI,yI,zI,RGB,nslices)
    pos = get(hf,'position');
    pos(3) = pos(3)*2;
    set(hf,'position',pos)
    saveas(hf,[prefix 'seg_overlay.png'])
    
    sliceView(xI,yI,zI,LRGB,nslices)
    saveas(hf,[prefix 'seg_only.png'])
    sliceView(xI,yI,zI,Js,nslices)
    saveas(hf,[prefix 'image_only.png'])
    
    
    %%
    % last write out Jdef and L as analyze
    % /*Acceptable values for datatype are*/
    % #define DT_NONE             0
    % #define DT_UNKNOWN          0    /*Unknown data type*/
    % #define DT_BINARY           1    /*Binary             ( 1 bit per voxel)*/
    % #define DT_UNSIGNED_CHAR    2    /*Unsigned character ( 8 bits per voxel)*/
    % #define DT_SIGNED_SHORT     4    /*Signed short       (16 bits per voxel)*/
    % #define DT_SIGNED_INT       8    /*Signed integer     (32 bits per voxel)*/
    % #define DT_FLOAT           16    /*Floating point     (32 bits per voxel)*/
    % #define DT_COMPLEX         32    /*Complex (64 bits per voxel; 2 floating point numbers)/*
    % #define DT_DOUBLE          64    /*Double precision   (64 bits per voxel)*/
    % #define DT_RGB            128    /*A Red-Green-Blue datatype*/
    % #define DT_ALL            255    /*Undocumented*/
    
    
    % write out Ji at this res ( in case the hig hres doesn't work)
    avw = avw_hdr_make;
    avw.hdr.dime.datatype = 16; % 16 bits FLOAT
    avw.hdr.dime.bitpix = 16;
    avw.hdr.dime.dim(2:4) = size(Ji);
    avw.hdr.dime.pixdim([3,2,4]) = dxJ;
    avw.img = Ji;
    avw_img_write(avw,[prefix 'target_to_atlas_low_res_pad.img'])
    
    
    
    if downloop == 1
        continue
    end
    %%
    
    % outputs for Vikram
    % we need to transform data to match the 10 micron atlas
    %template_name_10 = '/cis/home/dtward/Documents/ARA/Mouse_CCF/average_template_10.nrrd';
    %label_name_10 = '/cis/home/dtward/Documents/ARA/Mouse_CCF/annotation_10_2017.nrrd';
    %[L10,meta] = nrrdread(label_name_10);
    % below workaround because encountered  gunzip error 'unexpected end of file'
    %meta = nhdr_nrrd_read(label_name_10,true);
    %L10 =  meta.data
    %dxI10 = diag(sscanf(meta.spacedirections,'(%d,%d,%d) (%d,%d,%d) (%d,%d,%d)',[3,3]))';
    %nxI10 = [size(L10,2),size(L10,1),size(L10,3)];
    % hard coding the voxel size and image shape
    % for the 10um Allen Reference Atlas
%    dxI10 = [10 10 10]
%    nxI10 = [1320 800 1140]
%    xI10 = (0:nxI10(1)-1)*dxI10(1);
%    yI10 = (0:nxI10(2)-1)*dxI10(2);
%    zI10 = (0:nxI10(3)-1)*dxI10(3);
%    xI10 = xI10 - mean(xI10);
%    yI10 = yI10 - mean(yI10);
%    zI10 = zI10 - mean(zI10);
%    
%    
%    
%    
%    % reload and re downsample data
%    %%
%    % downsample to about same res as atlas
%    down = round(dxI10./dxJ0);
%    for f = 1 : length(info)
%        %disp(['File ' num2str(f) ' of ' num2str(length(info))])
%        J_ = double(imread(target_name,f));
%        if f == 1
%            nxJ0 = [size(J_,2),size(J_,1),length(info)];
%            nxJ = floor(nxJ0./down);
%            J = zeros(nxJ(2),nxJ(1),nxJ(3));
%        end
%        % downsample J_
%        Jd = zeros(nxJ(2),nxJ(1));
%        for i = 1 : down(1)
%            for j = 1 : down(2)
%                Jd = Jd + J_(i:down(2):down(2)*nxJ(2), j:down(1):down(1)*nxJ(1))/down(1)/down(2);
%            end
%        end
%        
%        slice = floor( (f-1)/down(3) ) + 1;
%        if slice > nxJ(3)
%            break;
%        end
%        J(:,:,slice) = J(:,:,slice) + Jd/down(3);
%        
%        if ~mod(f-1,10)
%            danfigure(1234);
%            imagesc(J(:,:,slice));
%            axis image
%            drawnow;
%        end
%    end
%    dxJ = dxJ0.*down;
%    xJ = (0:nxJ(1)-1)*dxJ(1);
%    yJ = (0:nxJ(2)-1)*dxJ(2);
%    zJ = (0:nxJ(3)-1)*dxJ(3);
%    
%    xJ = xJ - mean(xJ);
%    yJ = yJ - mean(yJ);
%    zJ = zJ - mean(zJ);
%    
%    xJp = [xJ(1)-dxJ(1), xJ, xJ(end)+dxJ(1)];
%    yJp = [yJ(1)-dxJ(2), yJ, yJ(end)+dxJ(2)];
%    zJp = [zJ(1)-dxJ(3), zJ, zJ(end)+dxJ(3)];
%    Jp = padarray(J,[1,1,1]);
%    
%    climJ = [min(J(:)),max(J(:))];
%    %%
%    % deformatlas target to atlas
%    % first thing is to upsample Aphi to Aphi10
%    % due to memory issues, I'm going to have to loop through slices,
%    % so this will be a bit slow
%    Jdef = zeros([nxI10(2) nxI10(1) nxI10(3)]);
%    Fx = griddedInterpolant({yI,xI,zI},Aphi1tinvx,'linear','nearest');
%    Fy = griddedInterpolant({yI,xI,zI},Aphi1tinvy,'linear','nearest');
%    Fz = griddedInterpolant({yI,xI,zI},Aphi1tinvz,'linear','nearest');
%    F = griddedInterpolant({yJp,xJp,zJp},Jp,'linear','nearest');
%    [XI10_,YI10_] = meshgrid(xI10,yI10);
%    for i = 1 : size(Jdef,3)
%        disp(['Applying deformation slice ' num2str(i) ' of ' num2str(size(Jdef,3))]);
%        
%        Aphi10x = Fx(YI10_,XI10_,ones(size(XI10_))*zI10(i));
%        Aphi10y = Fy(YI10_,XI10_,ones(size(XI10_))*zI10(i));
%        Aphi10z = Fz(YI10_,XI10_,ones(size(XI10_))*zI10(i));
%        
%        
%        Jdef(:,:,i) = F(Aphi10y,Aphi10x,Aphi10z);
%        
%        if ~mod(i,25)
%            danfigure(22993);
%            sliceView(xI,yI,zI,Jdef,5,climJ)
%            drawnow
%        end
%        
%    end
%    
%    %%
%    % last write out Jdef and L as analyze
%    % /*Acceptable values for datatype are*/
%    % #define DT_NONE             0
%    % #define DT_UNKNOWN          0    /*Unknown data type*/
%    % #define DT_BINARY           1    /*Binary             ( 1 bit per voxel)*/
%    % #define DT_UNSIGNED_CHAR    2    /*Unsigned character ( 8 bits per voxel)*/
%    % #define DT_SIGNED_SHORT     4    /*Signed short       (16 bits per voxel)*/
%    % #define DT_SIGNED_INT       8    /*Signed integer     (32 bits per voxel)*/
%    % #define DT_FLOAT           16    /*Floating point     (32 bits per voxel)*/
%    % #define DT_COMPLEX         32    /*Complex (64 bits per voxel; 2 floating point numbers)/*
%    % #define DT_DOUBLE          64    /*Double precision   (64 bits per voxel)*/
%    % #define DT_RGB            128    /*A Red-Green-Blue datatype*/
%    % #define DT_ALL            255    /*Undocumented*/
%    
%    
%    danfigure(22994);
%    avw = avw_hdr_make;
%    avw.hdr.dime.datatype = 4; % 16 bits
%    avw.hdr.dime.bitpix = 16;
%    avw.hdr.dime.dim(2:4) = size(Jdef);
%    avw.hdr.dime.pixdim([3,2,4]) = [10 10 10];
%    avw.img = Jdef;
%    avw_img_write(avw,[prefix 'target_to_atlas.img'])
    
    %end
    
    
    % avw.img = L10;
    % avw_img_write(avw,[prefix 'atlas_labels.img'])
    
    
end % of downloop

