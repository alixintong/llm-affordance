# llm_affordance

An object's affordance is its "opportunity for action", what a given organism can use that object for.

Understanding affordances is a useful test for language comprehension, as it typically requires knowledge of the world, actions, and individual words.

This study is interested in addressing whether multimodal language models can capture the affordances of an object, and its implications in whether artificial systems have some form of embodied cognition.

The following is an example of a data point.
Scenario: “After wading barefoot in the lake, Erik needed something to get dry. What would he use?”
1. Related object:[image of a towel]
2. Afforded object:[image of a shirt]
3. Non-afforded object:[image of glasses]
<img width="500" alt="affordance-example" src="https://github.com/alixintong/llm-affordance-draft/assets/93224235/fef816f8-a606-4b93-b36d-154dd927a440">

Additionally we prompt-engineered 2 methods (explicit, implicit) of asking the model to select the best image for a given scenario to test its understanding of affordance relationships. An example of an explicit version of the prompt vs its implicit version is as follows:
1. Explicit: “What would Erik use to dry himself?”
2. Implicit: “Erik would use this to dry himself”

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alixintong/llm_affordance.git

2. Navigate to the project directory:
   ```bash
   cd llm_affordance

3. Install the requirements:
   ```bash
   pip install -r requirements.txt

# Running Main Experiment
```bash
cd main_experiment
python src/main.py --dataset [dataset_name] --model [model_name] --relationship [relationship_type]
```
1. dataset: choose a dataset from ['natural','synthetic']
2. model: choose a model from ['ViT-B-32','ViT-L-14-336','ViT-H-14','ViT-g-14','ViT-bigG-14','ViT-L-14','imagebind']
3. relationship: running the main experiment (afforded vs non-afforded) or the follow-up manipulation check experiment (related vs non-afforded). Choose relationship from ['afforded','related']
