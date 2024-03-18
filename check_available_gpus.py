import GPUtil

def list_gpus_and_memory():
    # Attempt to get GPU information
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, Free Memory: {gpu.memoryFree}MB")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
list_gpus_and_memory()
