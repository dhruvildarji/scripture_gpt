
@echo off
set modelDir=%1

current_dir=%CD%\..\

cd %current_dir%\TensorRT-LLM\examples\gpt

python hf_gpt_convert.py -i %current_dir%\model\%modelDir% -o %current_dir%\model\%modelDir%\gpt_build --tensor-parallelism 1 --storage-type float16

python build.py --model_dir=%current_dir%\model\%modelDir%\gpt_build\1-gpu --use_gpt_attention_plugin --remove_input_padding --output_dir=%current_dir%\model\%modelDir%\engine_output
