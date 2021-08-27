CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the prostate
G_RGX = Case\d+_

TT_DATA = [('IMG', nii_transform2, False), ('GT', nii_gt_transform2, False), ('GT', nii_gt_transform2, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = data/prostate/SA

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/prostate/cesource/last.pkl

#run the main experiment
TRN = results/prostate/sfda

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-CSize.tar.gz

all: pack
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(TRN) $(INF_0) $(TRN_1) $(INF_1) $(TRN_2) $(TRN_3) $(TRN_4)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available

# first train on the source dataset only:
results/prostate/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "data/prostate/SB" \
	     --network UNet --model_weights="" --lr_decay 1 \
	    
# full supervision
results/prostate/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --lr_decay 1 \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/prostate/sfda: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/prostate/cesource/last.pkl" below):
results/prostate/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim  --batch_size 1  --l_rate 0 --model_weights="$(M_WEIGHTS_ul)" --pprint --n_epoch 1 --saveim True --entmap\

results/prostate/sfdaim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --saveim True --entmap --do_asd 1 --do_hd 1  --batch_size 1  --l_rate 0 --model_weights="results/prostate/sfda/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\


$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 4 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


