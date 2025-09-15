
BATCH_SIZE=64
LR=1e-4
EPOCHS=1
NUM_PROMPTS=500
PROMPT_DIM=768
WANDB_PROJECT="OS-Prompt"
GPU_IDS="0,1"
PATIENCE=5
SAVE_CRITERIA="f1_score"
SCHEDULER="cosine"
STEP_SIZE=10
GAMMA=0.1
MILESTONES="30 60 90"
LOSS_FUNCTION="cross_entropy"
IMG_SIZE=384
TRAIN_MODE=0
STEP_1_MODEL_PT_PATH="None"
MODEL_NAME="base_model"
SAVE_PATH="None"
WEIGHT_DECAY=1e-3
OPTIMIZER="adamw"


TRAIN_CSV_FILE="None" # This option is not used.
VAL_CSV_FILE="None" # This option is not used.
TEST_CSV_FILE="None" # This option is not used.
IMAGE_FOLDER="None" # This option is not used.
NUM_CLASSES=3   # This option is not used.

# 3) diabetic 
diabetic_TRAIN_CSV_FILE="./dataloader/diabetic/train.csv" 
diabetic_VAL_CSV_FILE="./dataloader/diabetic/val.csv"
diabetic_TEST_CSV_FILE="./dataloader/diabetic/test.csv"
diabetic_IMAGE_FOLDER="" #Please enter the diabetic image data path here.
NUM_CLASSES_2=3  


####################################
# 4) Python 
####################################
python3 main.py \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --num_prompts $NUM_PROMPTS \
    --prompt_dim $PROMPT_DIM \
    --wandb_project $WANDB_PROJECT \
    --gpu_ids $GPU_IDS \
    --patience $PATIENCE \
    --save_criteria $SAVE_CRITERIA \
    --scheduler $SCHEDULER \
    --step_size $STEP_SIZE \
    --gamma $GAMMA \
    --milestones $MILESTONES \
    --loss_function $LOSS_FUNCTION \
    --img_size $IMG_SIZE \
    --train_mode $TRAIN_MODE \
    --step_1_model_pt_path $STEP_1_MODEL_PT_PATH \
    --model_name $MODEL_NAME \
    --save_path $SAVE_PATH \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    \
    --train_csv_file $TRAIN_CSV_FILE \
    --val_csv_file $VAL_CSV_FILE \
    --test_csv_file $TEST_CSV_FILE \
    --image_folder $IMAGE_FOLDER \
    --num_classes $NUM_CLASSES \
    \
    --diabetic_train_csv_file $diabetic_TRAIN_CSV_FILE \
    --diabetic_val_csv_file $diabetic_VAL_CSV_FILE \
    --diabetic_test_csv_file $diabetic_TEST_CSV_FILE \
    --diabetic_image_folder $diabetic_IMAGE_FOLDER \
    --diabetic_classes $NUM_CLASSES_2 \