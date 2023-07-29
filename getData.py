# -*- coding: utf-8 -*-
import argparse
import logging
from libs.lottery import Lottery
from libs import utils

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description='Taiwan Lottery 台灣彩券爬蟲',
        epilog='https://github.com/stu01509/TaiwanLottery')

    parser.add_argument('game', nargs='?', default='',
                        help='爬取指定的彩券類型, 威力彩 大樂透 今彩539 雙贏彩')
    parser.add_argument('-b', '--back', default=0,
                        help='爬取幾個月前的資料')
    parser.add_argument('-t', '--time', default=0,
                        help='爬取指定年月份的資料 格式(YYYY-MM)')
    parser.add_argument('-o', '--output', action='store_true',
                        help='將爬取資料輸出成 json')
    parser.add_argument('-m', '--mix', action='store_true')

    args = parser.parse_args()
    logging.debug(args)

    lottery = Lottery()

    if (args.output == False and args.game == '' and args.back == 0  and args.time == 0):
        lottery.superLotto()
        lottery.lotto649()
    elif (args.output == True and args.game == '' and args.back == 0 and args.time == 0):
        lottery.superLotto(True, True)
        lottery.lotto649(True, True)
    elif (args.output == False and args.game != '' and args.back != 0 and args.time == 0):
        if (args.game == '威力彩'):
            lottery.superLottoBack(True, False, args.back)
        elif (args.game == '大樂透'):
            lottery.lotto649Back(True, False, args.back, args.mix)
    elif (args.output == True and args.game != '' and args.back != 0 and args.time == 0):
        if (args.game == '威力彩'):
            lottery.superLottoBack(True, True, args.back)
        elif (args.game == '大樂透'):
            lottery.lotto649Back(True, True, args.back, args.mix)
    elif (args.output == False and args.game != '' and args.back == 0 and args.time != 0):
        if (args.game == '威力彩'):
            lottery.superLotto(True, False, utils.convertToRepublicEraMonth(args.time))
        elif (args.game == '大樂透'):
            lottery.lotto649(True, False, utils.convertToRepublicEraMonth(args.time))
    elif (args.output == True and args.game != '' and args.back == 0 and args.time != 0):
        if (args.game == '威力彩'):
            lottery.superLotto(True, True, utils.convertToRepublicEraMonth(args.time))
        elif (args.game == '大樂透'):
            lottery.lotto649(True, True, utils.convertToRepublicEraMonth(args.time))


if __name__ == '__main__':
    main()
