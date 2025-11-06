# SLURM Guide for University Cluster

## Table of Contents
1. [Introduction](#introduction)
2. [Basic SLURM Commands](#basic-slurm-commands)
3. [Creating SLURM Job Scripts](#creating-slurm-job-scripts)
4. [Submitting and Managing Jobs](#submitting-and-managing-jobs)
5. [Examples for This Project](#examples-for-this-project)
6. [Common Issues and Tips](#common-issues-and-tips)

---

## Introduction

SLURM (Simple Linux Utility for Resource Management) is a job scheduler used on the university cluster. It manages compute resources and queues jobs for execution.

**Key Concepts:**
- **Node**: A physical computer in the cluster
- **Partition**: A group of nodes (e.g., CPU, GPU, high-memory)
- **Job**: A task you submit to run on the cluster
- **Allocation**: Resources assigned to your job

---

## Basic SLURM Commands

### Checking Cluster Status

```bash
# View available partitions and nodes
sinfo

# View node details
sinfo -N -l

# Check available GPUs (if applicable)
sinfo -o "%N %G"

# View your jobs
squeue --me

# View all jobs in the queue
squeue

# View detailed job information
scontrol show job <job_id>
```

### Managing Jobs on sbatch

```bash
# Submit a job
sbatch job_script.sh

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Hold a job (prevent it from starting)
scontrol hold <job_id>

# Release a held job
scontrol release <job_id>
```

### Job Information

```bash
# Check job status and resource usage
sacct -j <job_id>

# Detailed accounting information
sacct -j <job_id> --format=JobID,JobName,Partition,State,Start,End,Elapsed,MaxRSS,MaxVMSize

# View your recent jobs
sacct -u $USER --starttime today
```

---

## Creating SLURM Job Scripts

A SLURM job script is a bash script with special SLURM directives (lines starting with `#SBATCH`).

### Basic Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job           # Job name
#SBATCH --output=logs/%j.out        # Standard output (%j = job ID)
#SBATCH --error=logs/%j.err         # Standard error
#SBATCH --time=24:00:00             # Time limit (HH:MM:SS)
#SBATCH --mem=32G                   # Memory per node
#SBATCH --cpus-per-task=4           # Number of CPU cores
#SBATCH --partition=default         # Partition name
#SBATCH --mail-type=END,FAIL        # Email notifications
#SBATCH --mail-user=your@email.com  # Your email

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Your commands here
source /path/to/your/venv/bin/activate
python your_script.py


```

### GPU Job Template

```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --partition=gpu             # GPU partition

# Activate environment
source /path/to/your/venv/bin/activate

# Run your job
python train.py
```

### Common SBATCH Options

| Option | Description | Example |
|--------|-------------|---------|
| `--job-name` | Job name | `--job-name=eeg_training` |
| `--time` | Time limit | `--time=24:00:00` (24 hours) |
| `--mem` | Memory per node | `--mem=32G` |
| `--cpus-per-task` | CPU cores | `--cpus-per-task=8` |
| `--gres` | Generic resources (GPUs) | `--gres=gpu:2` (2 GPUs) |
| `--partition` | Partition to use | `--partition=gpu` |
| `--nodes` | Number of nodes | `--nodes=1` |
| `--ntasks` | Number of tasks | `--ntasks=1` |
| `--array` | Job array | `--array=1-10` |

---

## Submitting and Managing Jobs

### Submit a Job

```bash
# Submit a job script
sbatch job_script.sh

# Submit with custom parameters
sbatch --time=12:00:00 --mem=16G job_script.sh

# Submit and capture job ID
job_id=$(sbatch --parsable job_script.sh)
echo "Submitted job: $job_id"
```

### Monitor Jobs

```bash
# Watch your jobs (updates every 2 seconds)
watch -n 2 'squeue -u $USER'

# Check specific job
squeue -j <job_id>

# Check job output in real-time
tail -f logs/<job_id>.out
```

### Interactive Sessions

```bash
# Start interactive session
srun --pty --mem=16G --cpus-per-task=4 --time=2:00:00 bash

# Interactive session with GPU
srun --pty --gres=gpu:1 --mem=32G --time=4:00:00 bash

# Ask for specific node
srun --pty -p gpu.q -w <node_name> bash
```

---

## Common Issues and Tips

### Tips for Efficient Resource Usage

1. **Request appropriate resources**: Don't over-request memory or time
2. **Test with short jobs**: Use `--time=1:00:00` for testing
3. **Use job arrays**: Run multiple similar jobs efficiently
4. **Monitor resource usage**: Use `seff <job_id>` after jobs complete
5. **Create logs directory**: `mkdir -p logs` before submitting jobs

### Common Issues

#### Job Pending Forever
- Check partition availability: `sinfo`
- Reduce resource requests
- Check cluster policies: `scontrol show job <job_id>`

#### Out of Memory
```bash
# Check actual memory usage
sacct -j <job_id> --format=JobID,MaxRSS,ReqMem

# Increase memory in next submission
#SBATCH --mem=64G  # or higher
```

#### Job Time Limit
```bash
# Implement checkpointing in your code
# Resume from checkpoint if job times out
python train.py --resume-from-checkpoint path/to/checkpoint
```

#### Permission Denied
```bash
# Make script executable
chmod +x job_script.sh

# Check file permissions
ls -la job_script.sh
```

### Debugging Jobs

```bash
# View detailed error messages
cat logs/<job_id>.err

# Check job accounting
sacct -j <job_id> --format=ALL

# View node information
scontrol show node <node_name>
```

### Best Practices

1. **Always specify time limits**: Avoid default maximums
2. **Use meaningful job names**: Easy to identify in queue
3. **Organize output logs**: Use subdirectories or naming schemes
4. **Test interactively first**: Debug before batch submission
5. **Save checkpoints**: For long-running jobs
6. **Clean up old logs**: Prevent clutter
7. **Use dependencies**: Chain jobs with `--dependency`

### Job Dependencies

```bash
# Submit job and get ID
job1=$(sbatch --parsable job1.sh)

# Submit job2 that runs after job1 completes
sbatch --dependency=afterok:$job1 job2.sh

# Submit job3 that runs after job1 and job2 complete
sbatch --dependency=afterok:$job1:$job2 job3.sh
```

### Environment Variables

Useful SLURM environment variables available in your job:

- `$SLURM_JOB_ID` - Job ID
- `$SLURM_JOB_NAME` - Job name
- `$SLURM_NODELIST` - List of allocated nodes
- `$SLURM_CPUS_PER_TASK` - CPUs per task
- `$SLURM_MEM_PER_NODE` - Memory per node
- `$SLURM_ARRAY_TASK_ID` - Array task ID (for job arrays)

---

## Quick Reference

```bash
# Submit job
sbatch script.sh

# Check your jobs
squeue -u $USER

# Cancel job
scancel <job_id>

# Job details
scontrol show job <job_id>

# Interactive session
srun --pty --mem=16G --time=2:00:00 bash
```

