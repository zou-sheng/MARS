import os
import logging
import yaml
import math
from functools import reduce
from collections import defaultdict, OrderedDict
import subprocess
import random

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
                level=logging.INFO)
logger = logging.getLogger(__name__)

def run_timeloop(arch, prob, mapp, cwd=os.getcwd(), stdout=None, stderr=None):
    try:
        p = subprocess.check_call(['/home/mingchuan/Desktop/zousheng/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-model', str(arch), str(prob), str(mapp)], \
                                  cwd=cwd, stdout=stdout, stderr=stderr)
        logger.info('run_timeloop> timeloop-model {} {} {}'.format(arch, prob, mapp))
        return True
    except:
        return False

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
    return data

def get_factors(n):
    return list(reduce(list.__add__,
                        ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

def get_prime_factors(num):
    i = 2
    factors = defaultdict(int)
    while i * i <= num:
        while num % i == 0:
            factors[i] += 1
            num //= i
        i += 1
    if num > 1:
        factors[num] += 1
    return factors

def generate_factors_list(m, n):
    if n < 1:
        print("n 必须大于等于1")
        return []
    
    if m == 0:
        print("m 为0时，所有因数必须为0")
        return [0] * n
    
    if m == 1:
        return [1] * n
    
    factors = get_prime_factors(m)

    result = [1] * n  # 初始化结果列表为 n 个1
    random.shuffle(factors)  # 随机打乱因数
    
    for prime, count in factors.items():
        for _ in range(count):
            index = random.randint(0, n-1)
            result[index] *= prime
    
    
    return result

if __name__ == "__main__":
    print(get_factors(16))
    print(get_prime_factors(16).keys())
    print(generate_factors_list(15, 3))