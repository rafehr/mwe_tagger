from conllu import TokenList

def change_lextag_labels(sents: TokenList):
    """Changes the LEXTAG field of data in CONLLULEX format so that only
    Verbal Multiword-Expressions (VMWES) receive BIO2-style labels
    (but including the VMWE category) and all the other tokens 'O'.

    E.g. I~-V.VID-v.emotion becomes I-VID
    """
    vmwe_types = [
        'IAV', 'LVC.full', 'LVC.cause', 'VID', 'VPC.full', 'VPC.semi' 
    ]
    for sent in sents:
        curr_idx = None
        for tok in sent:
            # Check if token has an entry in the 'strong mwe' field
            if tok['smwe'] != '_':
                mwe_idx, comp_idx = map(int, tok['smwe'].split(':'))
                # Check whether MWE is a verbal one
                if any(vmwe in tok['lexcat'] for vmwe in vmwe_types):
                    vmwe_type = tok["lexcat"].replace("V.", "")
                    if comp_idx == 1:
                        # The first VMWE component gets the B-tag
                        tok['lextag'] = f'B-{vmwe_type}'
                        # Remember index of the current target VMWE
                        curr_idx = mwe_idx
                else:
                    # Give a MWE component the I-tag if it is part
                    # of the current target VMWE
                    if mwe_idx == curr_idx:
                        tok['lextag'] = f'I-{vmwe_type}' 
                    else:
                        # Tokens being part of a MWE, but not a verbal
                        # one are not considered
                        tok['lextag'] = 'O'
            else:
                tok['lextag'] = 'O'

def change_deprels(sents: TokenList):
    """Changes the dependency relations so that only the relation
    between the verb of a VPC and its particle stays the same. All
    the other relations are changed to a dummy label.
    """
    vpcs, deprels = [], set()
    for sent in sents:
        mwe_idx, vpc_idx = 1, 2 
        for tok in sent:
            change_label = True
            if tok['smwe'] != '_':
                mwe_idx, comp_idx = tok['smwe'].split(':')
            if 'VPC' in tok['lexcat']:
                vpc_idx = mwe_idx
                verb = tok['form']
                mwe_idx = -1
                continue
            if mwe_idx == vpc_idx and 'VPC' not in tok['lexcat']:
                vpcs.append((verb, tok['deprel'], tok['form']))
                deprels.add(tok['deprel'])
                mwe_idx, vpc_idx = 1, 2
                change_label = False
            if change_label:
                tok['deprel'] = 'None'
