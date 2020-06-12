maxNumCompThreads(36)
run('~/MBAC/registration/map_nonuniform_multiscale_v02_mouse_gauss_newton2.m');
atlas_prefix = './atlases/';
atlas_path = [atlas_prefix 'ara_annotation_10um.tif'];
atlas_voxel_size = [10.0, 10.0, 10.0]; % microns
output_path = [prefix 'labels_to_target_highres.img'];
down2 = [1,1,1];
nxJ0_ds = nxJ0./down2
dxJ0_ds = dxJ0.*down2
Aname = [prefix 'A.mat'];
vname = [prefix 'v.mat'];
save([prefix 'transform_params.mat'],'atlas_path','atlas_voxel_size','output_path','nxJ0','dxJ0','vname','Aname')
transform_data(atlas_path,atlas_voxel_size,Aname,vname,dxI,dxJ0_ds,nxJ0_ds,'target',output_path,'nearest')
transform_data(target_name,dxJ0,Aname,vname,dxI,atlas_voxel_size,[1320 800 1140],'target',output_path,'nearest')
