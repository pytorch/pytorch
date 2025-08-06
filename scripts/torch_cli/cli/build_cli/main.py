from cement import Controller
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

class BuildController(Controller):
    class Meta:
        label = "build"
        stacked_on = "base"
        stacked_type = "nested"
        description = "Build CLI group"
        
