import pandas as pd
import argparse
import os
import subprocess

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str,default='/home/cc/my_mounting_point/shared_data/datamirror/haodongw/T4V/dataset/test')
    parser.add_argument('--destination_path', type=str, default='/home/cc/T4V/dataset/test')
    parser.add_argument('--sample_path', type=str)
    args = parser.parse_args()

    samples = pd.read_pickle(args.sample_path)
    
    print("~~~~~~~~~~ Copying remote data to destination ~~~~~~~~~~~")
    
    for _, row in samples.iterrows():
        path = row['video_path']
        print(path)
        if not os.path.exists(path):
            parsed_path = os.path.split(path)
            print(parsed_path)
            further_parsed_path = list(os.path.split(parsed_path[0])) + [parsed_path[1]]
            if not os.path.exists(parsed_path[0]):
                os.makedirs(parsed_path[0])
            if any(h in args.source_path[:7] for h in ["http", "ftp"]):
                subprocess.run([
                    "wget",
                    f"{os.path.join(args.source_path,further_parsed_path[1],further_parsed_path[2])}",
                    f"--directory-prefix={os.path.join(args.destination_path, parsed_path[0])}"])
            else:
                subprocess.run(["cp",f"{os.path.join(args.source_path,further_parsed_path[1],further_parsed_path[2])}",f"{parsed_path[0]}"])
    
    print("~~~~~~~~~~ Remote data sucessfully fetched ~~~~~~~~~~~")


