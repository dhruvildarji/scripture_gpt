@echo off
set modelDir=%1
set input_text=%2

set current_dir=%CD%\..\

cd %current_dir%\TensorRT-LLM\examples

python run.py --input_text %input_text% --engine_dir=%current_dir%\models\%modelDir%\engine_output --max_output_len 128
