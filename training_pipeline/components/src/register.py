from azureml.core import Run
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='',help='Name of the model in Azure Model Management')
    parser.add_argument('--model_assets_path',type=str, default='outputs',help='Location of trained model.')
    parser.add_argument('--model_file_name',type=str, default='model.pth',help='Model file name')
    parser.add_argument('--label_file_name',type=str, default='labels.txt',help='Labels file name')

    args,unparsed = parser.parse_known_args()
    
    print('Model assets path is:',args.model_assets_path)
    print('Model name is:',args.model_name)
    print('Model file name:',args.model_file_name)
    print('Label file name:',args.label_file_name)

    run = Run.get_context()
   
    print('Experiment:',run.experiment)
    print('Run id:',run._root_run_id)

    pipeline_run = Run(run.experiment, run._root_run_id)
    pipeline_run.upload_file(os.path.join("outputs", args.model_name, args.model_file_name), os.path.join(args.model_assets_path, args.model_file_name))
    pipeline_run.upload_file(os.path.join("outputs", args.model_name, args.label_file_name), os.path.join(args.model_assets_path, args.label_file_name))

    tags = {
       "Source":"Register Model step"
    }

    model = pipeline_run.register_model(model_name=args.model_name, model_path='outputs/'+args.model_name, tags=tags)
       
    print('Model registered: {} \nModel Description: {} \nModel Version: {}'.format(model.name, model.description, model.version))