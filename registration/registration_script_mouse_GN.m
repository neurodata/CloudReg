run('~/MBAC/registration/map_nonuniform_multiscale_v02_mouse_gauss_newton.m');
atlas_prefix = './atlases/';
atlas_path = [atlas_prefix 'ara_annotation_10um.tif'];
atlas_voxel_size = [10.0, 10.0, 10.0]; % microns
output_path = [prefix 'labels_to_target_highres.img'];
down2 = round(atlas_voxel_size./dxJ0)
nxJ0_ds = nxJ0./down2
dxJ0_ds = dxJ0.*down2
[a,b,c] = fileparts(prefix);
vname = [a,filesep, b,['downloop_' num2str(downloop) '_'], c , 'v.mat']
Aname = [prefix 'A.mat'];
vname = [prefix 'v.mat'];
transform_data(atlas_path,atlas_voxel_size,Aname,vname,dxI,dxJ0_ds,nxJ0_ds,'target',output_path,'nearest')
