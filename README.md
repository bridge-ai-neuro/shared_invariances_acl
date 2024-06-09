## Shared Model Capabilities Revealed by Perturbed Examples 

<br>

### Key-Dependencies:

    1. torch 1.10.2
    2. textattack 0.3.8
    3. numpy 1.24.2
    4. pandas 1.5.3
    5. matplotlib 3.6.3
    6. seaborn 0.12.2

### Steps for Running Code
    1. Finetune models using ./scripts/train.sh
    2. Generate perturbations that are invariant with respect to a reference model using  ./scripts/generate_perturbations.sh
    3. Compute shared-capabilities between reference and target models using ./scripts/shared_capabilities.sh

### Acknowledgements
- This repo borrows code from the [textattack](https://github.com/QData/TextAttack/tree/master) library for generating perturbations.