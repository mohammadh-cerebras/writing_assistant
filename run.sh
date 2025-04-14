API=OPENAI
MODEL=gpt-4o-mini-2024-07-18
IFS='/' read -r -a array <<< $MODEL
MODEL_NAME=${array[-1]}
APPROACH=llm_mr
declare -a FILES=()


PRED_PATH=./predictions/literature_${API}_${MODEL_NAME}_title_and_related_work_and_contrib_specific_topic_step_by_step_relative_citing4.json
echo $PRED_PATH
python pred_llm.py \
    --api $API \
    --model $MODEL \
    --approach $APPROACH \
    --output_path $PRED_PATH 