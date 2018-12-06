## train
# BI, scale 2, 3, 4
##################################################################################################################################
# BI, scale 2, 3, 4
# DBDN2, input=48x48, output=96x96
LOG=experiment/DBDN2-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 th main.lua -scale 2 -netType DBDN -nFeat 64 -nFeaSDB 64 -nDenseBlock 16 -nDenseConv 8 -growthRate 64 -patchSize 96 -dataset flickr2k -datatype t7  -DownKernel BI -splitBatch 4 -trainOnly true 2>&1 | tee $LOG

# DBDN3, input=32x32, output=96x96
LOG=experiment/DBDN3-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 th main.lua -scale 3 -netType resnet_cu -nFeat 64 -nFeaSDB 64 -nDenseBlock 16 -nDenseConv 8 -growthRate 64 -patchSize 96 -dataset flickr2k -datatype t7  -DownKernel BI -splitBatch 4 -trainOnly true  -upsample deconv -preTrained ../experiment/model/DBDNx2.t7 2>&1 | tee $LOG

# DBDN4, input=32x32, output=128x128
LOG=experiment/DBDN4-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 th main.lua -scale 4 -netType resnet_cu -nFeat 64 -nFeaSDB 64 -nDenseBlock 16 -nDenseConv 8 -growthRate 64 -patchSize 128 -dataset flickr2k -datatype t7  -DownKernel BI -splitBatch 4 -trainOnly true  -preTrained ../experiment/model/DBDNx2.t7 2>&1 | tee $LOG

#######################################################################
#train DBDN+#######
LOG=experiment/DBDNplus2-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 th main.lua -scale 2 -netType resnet_cu -nFeat 64 -nFeaSDB 64 -nDenseBlock 16 -nDenseConv 8 -growthRate 64 -patchSize 96 -dataset flickr2k -datatype t7  -DownKernel BI -splitBatch 4 -trainOnly true -upsample espcnn -preTrained ../experiment/model/DBDNx2.t7 2>&1 | tee $LOG

# DBDN3, input=32x32, output=96x96
LOG=experiment/DBDNplus3-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 th main.lua -scale 3 -netType resnet_cu -nFeat 64 -nFeaSDB 64 -nDenseBlock 16 -nDenseConv 8 -growthRate 64 -patchSize 96 -dataset flickr2k -datatype t7  -DownKernel BI -splitBatch 4 -trainOnly true  -upsample espcnn -preTrained ../experiment/model/DBDNx3.t7 2>&1 | tee $LOG

# DBDN4, input=32x32, output=128x128
LOG=experiment/DBDNplus4-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 th main.lua -scale 4 -netType resnet_cu -nFeat 64 -nFeaSDB 64 -nDenseBlock 16 -nDenseConv 8 -growthRate 64 -patchSize 128 -dataset flickr2k -datatype t7  -DownKernel BI -splitBatch 4 -trainOnly true -upsample espcnn -preTrained ../experiment/model/DBDNx4.t7 2>&1 | tee $LOG





## other comments
# we use '-trainOnly true' to save GPU memory. Then we can train models with input patch size of 48x48. This allows us to keep the same input size as that in other methods (e.g.,EDSR). 
# In our arXiv paper, we reported the results with input patch size of 32x32. And the results with input size of 48x48 should be better.







