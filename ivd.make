CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#DEBUG = --debug

#the regex of the subjects in the target dataset
#for the ivd
G_RGX = Subj_\d+

TT_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
S_DATA = [('Wat', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the datasets
B_FOLD = data/ivd_transverse/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/ivd/cesource/last.pkl

#run the main experiment
TRN = results/ivd/sfda

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
results/ivd/cesource: OPT =  --target_losses="$(L_OR)" --target_folders="$(S_DATA)" --val_target_folders="$(S_DATA)" \
	     --network UNet --model_weights="" --lr_decay 1 --l_rate 5e-4 \
	    
# full supervision
results/ivd/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --model_weights="$(M_WEIGHTS_uce)" --lr_decay 1 \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/ivd/sfda: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/ivd.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
          #--saveim True --entmap --do_asd 1 --do_hd 1  \

#inference mode : saves the segmentation masks + entropy masks for a specific model saved as pkl file (ex. "$(M_WEIGHTS_ul)" below):
results/ivd/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim  --batch_size 1  --l_rate 0 --pprint --n_epoch 1 --saveim True --entmap \

$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 22 --n_class 2 --workdir $@_tmp --target_dataset "$(B_FOLD)"  \
                --grp_regex="$(G_RGX)"  --target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)"\
                --model_weights="$(M_WEIGHTS_ul)" --network=$(NET) \
                --lr_decay 0.9 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 3e-5 --lr_decay_epoch 50 --weight_decay 1e-4 $(OPT) $(DEBUG)\

	mv $@_tmp $@


