import logging
from datetime import datetime

class LocalTimeFormatter(logging.Formatter):
    """Formats the time in the local system timezone."""
    def formatTime(self, record, datefmt=None):
        # Create a datetime object from the log record's timestamp in the local timezone
        dt = datetime.fromtimestamp(record.created).astimezone()
        if datefmt:
            return dt.strftime(datefmt)
        else:
            # Replicating the default format of asctime, which is '%Y-%m-%d %H:%M:%S,%f' then stripped
            return dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

# Get root logger
root_logger = logging.getLogger()
# Remove any existing handlers to prevent duplicate logs
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Set up new handler with local time formatter
handler = logging.StreamHandler()
formatter = LocalTimeFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# Get the specific logger for the application
logger = logging.getLogger("uni-api")

# Set levels for other verbose loggers
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("watchfiles.main").setLevel(logging.CRITICAL)