'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import logging
import os

logger = logging.getLogger(__name__)

def mpi_discovery(distributed_port=29005, verbose=True):
    """
    Discovery MPI environment via mpi4py and map to relevant torch.distributed state
    """
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[-1]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        logger.info(
            "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'],
                    os.environ['LOCAL_RANK'],
                    os.environ['WORLD_SIZE'],
                    os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))

    if torch.distributed.is_initialized():
        assert torch.distributed.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(
            rank, torch.distributed.get_rank())
        assert torch.distributed.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
            world_size, torch.distributed.get_world_size())
