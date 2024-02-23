
@echo off
set folder=%1

cd D:\gpt2\TensorRT-LLM\examples\gpt

python hf_gpt_convert.py -i .\data\%folder% -o .\data\%folder%\gpt_build --tensor-parallelism 1 --storage-type float16

python build.py --model_dir=.\data\%folder%\gpt_build\1-gpu --use_gpt_attention_plugin --remove_input_padding --output_dir=.\data\%folder%\engine_output

python ..\run.py --input_text "000000" --engine_dir=.\data\%folder%\engine_output --max_output_len 128
