```bash
python train.py --attack_type pgd \
                --inference_output_dir /PATH/TO/INFERENCE_OUTPUT_DIR.pkl # path to save clean image inference result\
                --adv_inference_output_dir /PATH/TO/ADV_INFERENCE_OUTPUT_DIR.pkl # path to adv image inference result \
                --adv_image_dir /PATH/TO/ADV_IMAGE_DIR.pkl # path to save adv image \
                --hard-prompt # whether to use hard_prompt (optional) \
                --eval_only # if --hard_prompt, eval_only should be set \
                --depth 1 # 1 for KL loss if hard_prompt and other for cross entropy 
```
其他参数的均为默认
