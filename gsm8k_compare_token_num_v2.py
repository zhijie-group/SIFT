import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, help='Path to the input JSONL file')

input_file = parser.parse_args().base_path

output_file = input_file.split('predictions')[0] + 'final.jsonl'
output_acc_file = input_file.split('predictions')[0] + 'accuracy.json'

def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')

############################################ gsm8k_postprocess ############################################
def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]
###########################################################################################################

############################################ math_postprocess_v2 ##########################################
def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None

def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''),
                     (r'\ ', ''), (' ', ''), ('mbox', 'text'),
                     (',\\text{and}', ','), ('\\text{and}', ','),
                     ('\\text{m}', '\\text{}'), ('\\le', '<')]
    REMOVED_EXPRESSIONS = [
        'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
        'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
        'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals',
        'edges', 'students', 'childrentickets', 'multiples', '\\text{s}',
        '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}',
        '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}', r'\;', r',\!',
        '{,}', '"', '\\dots', '\n', '\r', '\f', 'am', 'pm', ':00'
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(\\text\{)\((.*?)\)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    assert '\n' not in final_answer
    assert '\r' not in final_answer
    assert '\f' not in final_answer
    if len(re.findall(r'finalansweris(.*)', final_answer)) > 0:
        final_answer = re.findall(r'finalansweris(.*)', final_answer)[-1]

    if len(re.findall(r'answer?is:?(.*)', final_answer)) > 0:
        final_answer = re.findall(r'answer?is:?(.*)', final_answer)[-1]

    if len(re.findall(r'oxed\{(.*?)\}', final_answer)) > 0:
        final_answer = re.findall(r'oxed\{(.*?)\}', final_answer)[-1]

    if len(re.findall(r'\$(.*?)\$', final_answer)) > 0:
        final_answer = re.findall(r'\$(.*?)\$', final_answer)[-1]
    final_answer = final_answer.strip()
    if 'rac' in final_answer and '\\frac' not in final_answer:
        final_answer = final_answer.replace('rac', '\\frac')

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer

def math_postprocess_v2(text: str) -> str:
    # print(text)
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    for maybe_ans in text.split('.'):
        # if 'final answer' in maybe_ans.lower():
        if re.search('final answer|answer is', maybe_ans.lower()):
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split('.')[0])
###########################################################################################################

class BaseEvaluator:

    def __init__(self) -> None:
        pass

    def score(self):
        raise NotImplementedError("Method hasn't been implemented yet")

    @staticmethod
    def is_num_equal(predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        else:
            return

class Gsm8kEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - float(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        correct = 0
        count = 0
        details = []

        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)

        accuracy = 100 * correct / count
        result = {'accuracy': accuracy, 'details': details}
        return result

class MATHEvaluator(BaseEvaluator):

    def __init__(self, version='v1'):
        assert version in ['v1', 'v2']
        self.version = version

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []

        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equiv(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)

        result = {'accuracy': 100 * correct / count, 'details': details}
        return result

    def _fix_fracs(self, string):
        substrs = string.split('\\frac')
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += '\\frac'
                if len(substr) > 0 and substr[0] == '{':
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except AssertionError:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != '{':
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}{' + b + '}' + post_substr
                        else:
                            new_str += '{' + a + '}{' + b + '}'
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += '{' + a + '}' + b + post_substr
                        else:
                            new_str += '{' + a + '}' + b
        string = new_str
        return string

    def _fix_a_slash_b(self, string):
        if len(string.split('/')) != 2:
            return string
        a = string.split('/')[0]
        b = string.split('/')[1]
        try:
            a = int(a)
            b = int(b)
            assert string == '{}/{}'.format(a, b)
            new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
            return new_string
        except AssertionError:
            return string

    def _remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing
        # units
        if '\\text{ ' in string:
            splits = string.split('\\text{ ')
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(self, string):
        if '\\sqrt' not in string:
            return string
        splits = string.split('\\sqrt')
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != '{':
                a = split[0]
                new_substr = '\\sqrt{' + a + '}' + split[1:]
            else:
                new_substr = '\\sqrt' + split
            new_string += new_substr
        return new_string

    def _fix_sqrt_v2(self, string):
        _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
        return _string

    def _strip_string(self, string):
        # linebreaks
        string = string.replace('\n', '')

        # remove inverse spaces
        string = string.replace('\\!', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')

        # remove units (on the right)
        string = self._remove_right_units(string)

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')
        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        # fix sqrt3 --> sqrt{3}
        string = self._fix_sqrt(string)

        # remove spaces
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works
        # with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == '0.5':
            string = '\\frac{1}{2}'

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix
        # in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def _strip_string_v2(self, string):
        string = str(string).strip()
        # linebreaks
        string = string.replace('\n', '')

        # right "."
        string = string.rstrip('.')

        # remove inverse spaces
        string = string.replace('\\!', '')
        string = string.replace('\\ ', '')

        # replace \\ with \
        string = string.replace('\\\\', '\\')
        string = string.replace('\\\\', '\\')

        # replace tfrac and dfrac with frac
        string = string.replace('tfrac', 'frac')
        string = string.replace('dfrac', 'frac')

        # remove \left and \right
        string = string.replace('\\left', '')
        string = string.replace('\\right', '')

        # Remove unit: miles, dollars if after is not none
        _string = re.sub(r'\\text{.*?}$', '', string).strip()
        if _string != '' and _string != string:
            string = _string

        # Remove circ (degrees)
        string = string.replace('^{\\circ}', '')
        string = string.replace('^\\circ', '')

        # remove dollar signs
        string = string.replace('\\$', '')
        string = string.replace('$', '')

        string = string.replace('\\text', '')
        string = string.replace('x\\in', '')

        # remove percentage
        string = string.replace('\\%', '')
        string = string.replace('\%', '')  # noqa: W605
        string = string.replace('%', '')

        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        string = string.replace(' .', ' 0.')
        string = string.replace('{.', '{0.')

        # cdot
        string = string.replace('\\cdot', '')

        # inf
        string = string.replace('infinity', '\\infty')
        if '\\infty' not in string:
            string = string.replace('inf', '\\infty')
        string = string.replace('+\\inity', '\\infty')

        # and
        string = string.replace('and', '')
        string = string.replace('\\mathbf', '')

        # use regex to remove \mbox{...}
        string = re.sub(r'\\mbox{.*?}', '', string)

        # quote
        string.replace("'", '')
        string.replace('"', '')

        # i, j
        if 'j' in string and 'i' not in string:
            string = string.replace('j', 'i')

        # replace a.000b where b is not number or b is end, with ab, use regex
        string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
        string = re.sub(r'(\d+)\.0+$', r'\1', string)

        # if empty, return empty string
        if len(string) == 0:
            return string
        if string[0] == '.':
            string = '0' + string

        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(string.split('=')) == 2:
            if len(string.split('=')[0]) <= 2:
                string = string.split('=')[1]

        string = self._fix_sqrt_v2(string)
        string = string.replace(' ', '')

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
        # Even works with \frac1{72} (but not \frac{72}1).
        # Also does a/b --> \\frac{a}{b}
        string = self._fix_fracs(string)

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
        # cases fix in case the model output is X/Y
        string = self._fix_a_slash_b(string)

        return string

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print('WARNING: Both None')
            return True
        if str1 is None or str2 is None:
            return False

        if self.version == 'v1':
            strip_string_func = self._strip_string
        elif self.version == 'v2':
            strip_string_func = self._strip_string_v2
        else:
            raise NotImplementedError

        try:
            ss1 = strip_string_func(str1)
            ss2 = strip_string_func(str2)
            if verbose:
                print(ss1, ss2)
            if ss1 == ss2:
                return True
            ss1 = normalize_final_answer(ss1)
            ss2 = normalize_final_answer(ss2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        try:
            ss1 = normalize_final_answer(str1)
            ss2 = normalize_final_answer(str2)
            if ss1 == ss2:
                return True
        except Exception:
            pass

        return str1 == str2

# Read the JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

evaluator = Gsm8kEvaluator()
evaluator_v2 = MATHEvaluator(version='v2')

predictions = []
references = []

# Open the output file
with open(output_file, 'w') as out_f:
    # Iterate over each item in the JSON data
    for key in data:
        # if key == "400":
        #     break
        item = data[key]
        # Extract the question
        question = item['origin_prompt'][0]['prompt']
        # Extract the prediction
        prediction = item['prediction']
        # Extract the gold (reference) answer
        gold = item['gold']
        total_token_len = item.get('total_token_len', None)
        # Process the prediction to extract the final answer
        pred = gsm8k_postprocess(prediction)
        pred_v2 = math_postprocess_v2(prediction)
        # Process the gold to extract the final answer
        answer = gsm8k_dataset_postprocess(gold)
        # Append to predictions and references lists
        predictions.append(pred)
        references.append(answer)
        # Determine if the prediction is correct
        is_correct = evaluator.is_equal(pred, answer)
        is_correct_v2 = evaluator_v2.is_equiv(pred_v2, answer)
        is_correct_final = is_correct or is_correct_v2
        # Create the output dictionary
        output_dict = {
            'question': question,
            'prediction': prediction,
            'gold': gold,
            'pred_v1': pred,
            'pred_v2': pred_v2,
            'answer': answer,
            'is_correct_v1': is_correct,
            'is_correct_v2': is_correct_v2,
            'is_correct_final': is_correct_final,
            'total_token_len': total_token_len,
            'input_token_len': item.get('input_token_len', None)
        }
        # Write the output dictionary as a JSON line
        out_f.write(json.dumps(output_dict) + '\n')

total_num = 0
is_correct_v1_num = 0
is_correct_v2_num = 0
is_correct_final_num = 0
total_token_lens = 0
input_token_lens = 0

with open(output_file, 'r') as out_f:
    for line in out_f:
        obj = json.loads(line)
        if obj['is_correct_v1']:
            is_correct_v1_num += 1
        if obj['is_correct_v2']:
            is_correct_v2_num += 1
        if obj['is_correct_final']:
            is_correct_final_num += 1
        total_num += 1
        total_token_lens += obj['total_token_len']
        input_token_lens += obj['input_token_len']

# Compute accuracy
accuracy = 100 * is_correct_v1_num / total_num
accuracy_v2 = 100 * is_correct_v2_num / total_num
accuracy_final = 100 * is_correct_final_num / total_num

print(f"Accuracy v1: {accuracy:.2f}%\nAccuracy v2: {accuracy_v2:.2f}%\nAccuracy final: {accuracy_final:.2f}%")
print('*'*20)
print(f"Current Num: {total_num}\nCorrect Num v1: {is_correct_v1_num}\nCorrect Num v2: {is_correct_v2_num}\nCorrect Num final: {is_correct_final_num}")
print('*'*20)
print(f"Total Token Num: {total_token_lens}")

# Write the accuracy to a JSON file
accuracy_dict = {
    'accuracy_v1': accuracy,
    'accuracy_v2': accuracy_v2,
    'accuracy_final': accuracy_final,
    'total_token_num': total_token_lens,
    'input_token_num': input_token_lens,
    'output_token_num': total_token_lens - input_token_lens
}

with open(output_acc_file, 'w') as out_f:
    json.dump(accuracy_dict, out_f, indent=4)
