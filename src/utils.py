import logging

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