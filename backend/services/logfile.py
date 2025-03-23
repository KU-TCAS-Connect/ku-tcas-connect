import os    

def save_log_infile(filename, content:list):
    if not os.path.exists(filename):
        open(filename, 'w', encoding="utf-8").close()
        
    with open(f"{filename}", "a",  encoding="utf-8") as file:
        for item in content:
            if type(item) == str:
                file.write(item)
                print(item)
            if type(item) == list:
                for i in item:
                    file.write(i)
                    print(i)
