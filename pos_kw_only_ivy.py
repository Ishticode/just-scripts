import ivy
import inspect
import numpy as np
import ivy.functional.backends.jax
import ivy.functional.backends.torch
import ivy.functional.backends.tensorflow

ivy_fw_dicts = [(ivy.functional.backends.jax.__dict__.copy(), "jax"),
                (ivy.functional.backends.torch.__dict__.copy(), "torch"),
                (ivy.functional.backends.tensorflow.__dict__.copy(), "tensorflow"),
                (ivy.functional.backends.numpy.__dict__.copy(), "numpy"),
                (ivy.functional.__dict__.copy(), "ivy")]
ivy_dict_copy = ivy.__dict__.copy()
for en, val in ivy_dict_copy.items():
    if isinstance(val, type):
        continue
    if callable(val):
        fws_kw_dict = {}
        for ivy_dict_copy_fw, fw in ivy_fw_dicts:

            try:
                val = ivy_dict_copy_fw[en]
            except KeyError:
                val = ivy_dict_copy[en]
            sig = inspect.signature(val)

            fws_kw_dict[fw] = [0, 0, 0]
            for param in sig.parameters.copy().values():
                if param.kind == param.POSITIONAL_ONLY:
                    fws_kw_dict[fw][0] += 1
                if param.kind == param.POSITIONAL_OR_KEYWORD:
                    fws_kw_dict[fw][1] += 1
                if param.kind == param.KEYWORD_ONLY:
                    fws_kw_dict[fw][2] += 1

        L = [x for x in fws_kw_dict.values()]
        out = (np.diff(np.vstack(L).reshape(len(L), -1), axis=0) == 0).all()
        if not out:
            print(en, fws_kw_dict)
