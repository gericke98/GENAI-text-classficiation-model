"File to create tar input to the model and analyze the best option for deployment in terms of latency and cost"
import json
import tarfile

input_data = [
    {"inputs": "Tesla stock bounces back after earnings"},
    {"inputs": "New treatment for adhd discovered"},
    {"inputs": "Life found at Mars"},
    {"inputs": "Star Wars remains top rated movie"},
]

# Transform into jsn file
def create_json_files(data):
    for i,d in enumerate(data):
        print(i)
        filename = f'input{i+1}.json'
        with open(filename,'w') as f:
            json.dump(d,f,indent=4)




# Compress all json files
def create_tar_file(input_files,output_filename = 'inputs.tar.gz'):
    with tarfile.open(output_filename,'w:gz') as tar:
        for file in input_files:
            tar.add(file)



def main():
    print("Creating json files...")
    create_json_files(input_data)
    input_files = [f'input{i+1}.json' for i in range(len(input_data))]
    create_tar_file(input_files)


if __name__ == '__main__':
    main()