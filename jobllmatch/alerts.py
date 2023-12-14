import logging
import os
import sys

import requests

server = "t.me/bart_finetune_alert_bot"
code = os.environ["TELEGRAM_API_KEY"]


def mes_done_device(messeage="done", code=None, chat_id=None):
    messeage = "{} : ".format(server) + messeage
    code = code
    if code is not None and chat_id is not None:
        teleurl = "https://api.telegram.org/bot" + code + "/sendMessage"
        params = {"chat_id": chat_id, "text": messeage}
        res = requests.get(teleurl, params=params)
        if res.status_code == 200:
            print("Telegram notification sent successfully.")
        else:
            print("Failed to send Telegram notification.")


def make_logger(name=None, logfilename="all"):
    # make 1 logger instance
    logging.basicConfig(
        format="[%(asctime)s] | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] | %(levelname)s | %(name)s | %(message)s"
    )
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logfile = "./logs/{}.log".format(logfilename)
    file_handler = logging.FileHandler(filename=logfile)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.info('logs will be written to [{}]'.format(logfile))
    return logger


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        mes_done_device(
            "Done device {}".format(sys.argv[1]), code=code, chat_id="your_chat_id"
        )
    else:
        mes_done_device("Done..", code=code, chat_id="your_chat_id")
