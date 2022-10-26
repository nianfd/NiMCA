suffix=data0_2077_441k_10s_gt4
label_process_filename=annotate.scp
data_dir=/home/momozyc/data/aqe/wav

data=${data_dir}/$suffix
label=${data_dir}/${label_process_filename}

feat_dir=joint_feat_${suffix}
dst_dir=data_${suffix}
nj=20
noise_dir=/home/momozyc/project/aqe/feature/tools/noise/noise_${suffix}  #不需要
denoise=denoise_${suffix}

#python make_fake_label.py $data $label 
#python make_label.py $data $label label_aug_momo_scr1_vol_batch2.scp 

cd tools
#bash denoise_para.sh $data $denoise $noise_dir $nj

cd ..
bash tools/feat_extact_joint.sh $data $nj $dst_dir $feat_dir $noise_dir

#link_dir=$feat_dir
#python make_feats.py $label $label_process_filename  $feat_dir 

#cd ../merge_vbase_fix_v15_wb
#ln -s ../feature/$feat_dir $feat_dir

