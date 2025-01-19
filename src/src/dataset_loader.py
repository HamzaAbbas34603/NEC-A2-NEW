# Dataset loading module
def parse_dataset(file_path):
    """
    Parse the Job Shop dataset from a file.
    Args:
        file_path (str): Path to the dataset file.
    Returns:
        list: A list of jobs, where each job is a list of (machine, time) tuples.
    """
    jobs = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                tasks = line.split()
                job = [(int(tasks[i]), int(tasks[i + 1])) for i in range(0, len(tasks), 2)]
                jobs.append(job)
    return jobs
