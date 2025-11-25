import logging
from ucimlrepo import fetch_ucirepo
import pandas
from BinaryClassifier import BinaryClassifier
from MulticlassClassifier import MulticlassClassifier

# Global logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    # fetch dataset 
    steel_plates_faults = fetch_ucirepo(id=198) 
    
    # data (as pandas dataframes) 
    X = steel_plates_faults.data.features 
    y = steel_plates_faults.data.targets 
    
    # metadata 
    print(steel_plates_faults.metadata)
    
    # variable information 
    print(steel_plates_faults.variables)

def main():
    binclass = BinaryClassifier()
    multiclass = MulticlassClassifier()
    load_data()

if __name__ == "__main__":
    main()