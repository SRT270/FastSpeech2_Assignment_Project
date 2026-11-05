import importlib.util, sys
spec = importlib.util.spec_from_file_location('test_proj', r'C:/Users/sneha/Downloads/FastSpeech2_Assignment_Project/FastSpeech2_Assignment_Project/tests/test_project.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in dir(mod):
    if name.startswith('test_'):
        func = getattr(mod, name)
        print(f'Running {name}...')
        func()
print('All tests passed')

