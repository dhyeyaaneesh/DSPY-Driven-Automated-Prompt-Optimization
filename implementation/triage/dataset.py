"""Synthetic, de-identified symptom-triage dataset.

Every case is generated from a hand-written *scenario bank*: each scenario is a
clinical presentation with a fixed reference urgency level, the red flags that
justify it, and several lay/clinical phrasings. Cases are rendered in three
narrative styles (clinical note, first-person patient message, caregiver
message) with random demographics, benign comorbidities and irrelevant
distractor sentences, so the model has to find the clinically decisive detail.

No real patient data is used. The reference labels encode common
safety-netting rules (NHS 111 / ESI-style red flags) and should be reviewed by
a clinician before any claim beyond "controlled benchmark" is made.

Splits are *scenario-grouped* by default: a scenario's cases never appear in
more than one split, so few-shot demos cannot be near-copies of test cases.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import dspy

from .labels import URGENCY_LEVELS

E, U, N, S = "EMERGENCY", "URGENT", "NON_URGENT", "SELF_CARE"


@dataclass
class Scenario:
    sid: str
    complaint: str
    level: str
    core: list[str]                      # alternative phrasings of the decisive presentation
    durations: list[str]
    extras: list[str] = field(default_factory=list)       # associated features (consistent with level)
    red_flags: list[str] = field(default_factory=list)    # gold red flags (empty for low acuity)
    ages: tuple[int, int] = (18, 80)
    sex: str = "any"                     # any | F | M
    history: list[str] | None = None     # scenario-specific history that matters (else random benign)


def _s(*args, **kw) -> Scenario:
    return Scenario(*args, **kw)


# --------------------------------------------------------------------------------------
# Scenario bank  (sid, complaint, level, core phrasings, durations, extras, red flags, ...)
# --------------------------------------------------------------------------------------
SCENARIOS: list[Scenario] = [
    # ---- chest pain ----
    _s("cp_acs", "chest pain", E,
       ["crushing pressure in the centre of the chest spreading to the left arm and jaw",
        "a heavy, squeezing chest pain that goes into the left arm"],
       ["for the last 40 minutes", "that started about half an hour ago while resting"],
       ["sweating heavily", "feeling sick", "short of breath"],
       ["central crushing chest pain", "radiation to arm/jaw", "diaphoresis"], ages=(45, 82)),
    _s("cp_angina", "chest pain", U,
       ["tightness in the chest that comes on when walking uphill and settles after a few minutes of rest",
        "a pressure feeling in the chest with exertion that eases when stopping"],
       ["for the past week", "over the last 5 days, getting a bit more frequent"],
       ["no pain at rest", "a family history of heart disease"],
       ["new exertional chest pain"], ages=(45, 78)),
    _s("cp_msk", "chest pain", N,
       ["sharp pain on the right side of the chest wall that is worse when pressing on the spot or twisting",
        "a sore spot on the chest wall after moving furniture, tender to touch"],
       ["for 6 days", "since last weekend"],
       ["no shortness of breath", "pain does not change with exertion"], ages=(20, 60)),
    _s("cp_reflux", "heartburn", S,
       ["a mild burning feeling behind the breastbone after a large spicy dinner, which went away after an antacid",
        "brief heartburn after a heavy late meal, relieved by antacid tablets"],
       ["last night, now gone", "once yesterday evening"],
       ["no pain on exertion", "eating and drinking normally"], ages=(20, 55)),

    # ---- headache ----
    _s("ha_sah", "headache", E,
       ["a sudden, explosive headache that reached full intensity within seconds, the worst ever experienced",
        "a thunderclap headache that came on instantly, like being hit on the back of the head"],
       ["starting 2 hours ago", "that began suddenly this afternoon"],
       ["vomiting", "neck stiffness"],
       ["thunderclap onset", "worst headache of life"], ages=(25, 70)),
    _s("ha_mening", "headache with fever", E,
       ["a severe headache with high fever, a stiff neck and a purple rash that does not fade under a glass",
        "fever, headache, pain bending the neck and dark spots on the skin that don't blanch"],
       ["since this morning", "developing over the last 12 hours"],
       ["light hurts the eyes", "increasingly drowsy"],
       ["non-blanching rash", "neck stiffness with fever"], ages=(16, 30)),
    _s("ha_gca", "headache", U,
       ["a new headache over the temple with a tender scalp and aching in the jaw when chewing",
        "new one-sided temple headache, sore scalp when brushing hair and jaw pain on chewing"],
       ["for 10 days", "over the past 2 weeks"],
       ["feeling generally unwell", "no vision change yet"],
       ["new headache age >50 with jaw claudication"], ages=(60, 85)),
    _s("ha_tension", "headache", N,
       ["recurring band-like headaches across the forehead, a few times a week",
        "frequent dull pressure headaches that come and go"],
       ["for about 3 months", "on and off since the start of the year"],
       ["no vomiting or vision change", "worse with work stress"], ages=(18, 50)),
    _s("ha_mild", "headache", S,
       ["a mild headache after a night of poor sleep, eased by paracetamol",
        "a dull headache after a long day at the screen that settled with rest and water"],
       ["since this morning", "today"],
       ["no fever", "no other symptoms"], ages=(18, 50)),

    # ---- breathing ----
    _s("sob_severe", "breathing difficulty", E,
       ["severe difficulty breathing, unable to finish a sentence, with lips turning bluish",
        "struggling to breathe, can only say a few words at a time, lips look blue"],
       ["for the last 20 minutes", "getting worse over the past hour"],
       ["using neck muscles to breathe", "confused"],
       ["cannot speak in full sentences", "cyanosis"]),
    _s("sob_asthma", "asthma flare", U,
       ["an asthma flare with wheezing, needing the reliever inhaler every 3 to 4 hours, which helps partly",
        "worsening wheeze and chest tightness, using the blue inhaler much more than usual"],
       ["for 2 days", "since a cold started 3 days ago"],
       ["can still speak in full sentences", "waking at night coughing"],
       ["escalating reliever use"], ages=(12, 60)),
    _s("sob_gradual", "breathlessness", N,
       ["gradually noticing breathlessness when climbing two flights of stairs, fine at rest",
        "slowly getting more out of breath with exercise"],
       ["over several months", "for about 4 months"],
       ["no chest pain", "no cough"], ages=(35, 70)),

    # ---- abdominal ----
    _s("ab_gibleed", "vomiting blood", E,
       ["vomiting bright red blood and passing black, tarry stools",
        "throwing up blood twice and having black sticky stools"],
       ["since this morning", "over the last few hours"],
       ["feeling faint when standing", "pale and clammy"],
       ["haematemesis", "melaena", "signs of shock"], ages=(30, 85)),
    _s("ab_append", "abdominal pain", U,
       ["pain that started around the belly button and has moved to the lower right side, worse when walking or coughing",
        "tummy pain that shifted to the lower right and hurts more with every bump in the car"],
       ["since yesterday evening", "for about 18 hours"],
       ["mild fever", "off food", "nausea"],
       ["migratory RLQ pain", "fever"], ages=(10, 40)),
    _s("ab_ibs", "abdominal discomfort", N,
       ["recurring bloating and cramping that eases after opening the bowels",
        "intermittent tummy cramps and bloating linked to meals"],
       ["for around 6 months", "on and off for most of the year"],
       ["no weight loss", "no blood in stools"], ages=(18, 45)),
    _s("ab_mild", "stomach upset", S,
       ["mild stomach cramps and loose stools after eating at a street stall",
        "a slightly upset stomach with two loose stools"],
       ["since yesterday", "for 1 day"],
       ["drinking fluids well", "no blood in stool", "no fever"], ages=(18, 60)),

    # ---- neuro ----
    _s("neuro_stroke", "weakness", E,
       ["sudden drooping of one side of the face, weakness in the right arm and slurred speech",
        "the face suddenly looks lopsided, cannot lift one arm properly and words are coming out jumbled"],
       ["that started 1 hour ago", "noticed suddenly 45 minutes ago"],
       ["no headache", "family noticed it"],
       ["FAST positive (face, arm, speech)"], ages=(50, 88)),
    _s("neuro_tia", "temporary vision loss", U,
       ["a brief episode of losing vision in one eye, like a curtain coming down, that fully resolved after 5 minutes",
        "a short spell of numbness in one hand and difficulty finding words that went away completely after 10 minutes"],
       ["yesterday", "earlier today"],
       ["completely back to normal now", "has high blood pressure"],
       ["transient focal neurology (possible TIA)"], ages=(50, 85)),
    _s("neuro_seizure_long", "seizure", E,
       ["a fit with jerking of all limbs that has been going on for more than 5 minutes",
        "a seizure that started 6 minutes ago and has not stopped"],
       ["right now", "currently ongoing"],
       ["not responding", "has epilepsy"],
       ["seizure >5 minutes (status epilepticus)"], ages=(5, 70)),
    _s("neuro_seizure_first", "first seizure", U,
       ["a first-ever seizure lasting about one minute, now awake and talking normally",
        "a single shaking episode with loss of consciousness for under a minute, never happened before, now fully recovered"],
       ["this morning", "about 3 hours ago"],
       ["tired but alert", "bit tongue"],
       ["first unprovoked seizure"], ages=(16, 60)),
    _s("neuro_bppv", "dizziness", N,
       ["brief spinning episodes lasting under a minute when rolling over in bed or looking up",
        "short bursts of the room spinning when turning the head quickly"],
       ["for 2 weeks", "on and off for 10 days"],
       ["no hearing loss", "no weakness or speech problems"], ages=(40, 75)),
    _s("neuro_postural", "light-headedness", S,
       ["feeling light-headed for a few seconds after standing up quickly from the sofa, then fine",
        "a moment of dizziness on getting up too fast that passed straight away"],
       ["once this morning", "today"],
       ["drinking less water than usual in the heat", "no fainting"], ages=(18, 45)),

    # ---- fever / infection ----
    _s("fev_sepsis", "fever and confusion", E,
       ["high fever with new confusion, very fast breathing and cold, blotchy skin",
        "a fever with shivering, now muddled and breathing very fast, skin mottled"],
       ["getting worse over 12 hours", "since last night"],
       ["has not passed urine all day", "recent urine infection"],
       ["new confusion", "tachypnoea", "mottled skin (possible sepsis)"], ages=(55, 90)),
    _s("fev_pyelo", "fever with urinary symptoms", U,
       ["fever of 39.4 °C, burning when passing urine and pain in the side of the back",
        "high temperature, stinging urine and a deep ache in the left flank"],
       ["for 1 day", "since yesterday"],
       ["shivering episodes", "still able to keep fluids down"],
       ["fever with flank pain (possible pyelonephritis)"], ages=(18, 65), sex="F"),
    _s("fev_cold", "cold symptoms", S,
       ["a runny nose, mild sore throat and temperature of 37.9 °C",
        "a blocked nose, sneezing and a slight temperature"],
       ["for 2 days", "since the weekend"],
       ["eating and drinking normally", "able to work from home"], ages=(18, 60)),
    _s("fev_infant", "fever in infant", E,
       ["a temperature of 38.3 °C in a baby who is feeding less than usual",
        "a fever of 38.5 °C measured twice"],
       ["since this evening", "for the last few hours"],
       ["more sleepy than usual"],
       ["fever in an infant under 3 months"], ages=(0, 0)),
    _s("fev_toddler", "fever in toddler", U,
       ["a fever around 39 °C that has not improved after 3 days, pulling at one ear",
        "a high temperature for 72 hours with ear pain, not settling with paracetamol"],
       ["for 3 days", "since Monday"],
       ["still drinking", "wet nappies as normal"],
       ["persistent fever >72 h in a young child"], ages=(1, 3)),
    _s("fev_vacc", "fever after vaccination", S,
       ["a low-grade temperature of 37.8 °C the day after routine vaccinations, playing and drinking normally",
        "a slight fever and a sore leg where the jab was given, otherwise happy"],
       ["since yesterday", "for 1 day"],
       ["wet nappies as normal"], ages=(1, 4)),
    _s("uti_simple", "urinary symptoms", N,
       ["burning when passing urine and needing to go more often, with no fever or back pain",
        "stinging on urination and frequency, otherwise well"],
       ["for 1 day", "since yesterday morning"],
       ["no blood in urine", "not pregnant"], ages=(18, 50), sex="F"),
    _s("cellulitis", "spreading redness", U,
       ["a hot, red, swollen area on the lower leg that has spread noticeably since yesterday, with fever",
        "a patch of red, painful skin on the shin that is getting bigger by the hour and a temperature"],
       ["for 2 days", "since yesterday"],
       ["a small cut there last week"],
       ["rapidly spreading skin infection with fever"], ages=(25, 80)),

    # ---- allergy / skin ----
    _s("anaphylaxis", "allergic reaction", E,
       ["swelling of the lips and tongue, wheezing and a widespread itchy rash after eating peanuts",
        "throat feels tight, lips swelling and hives all over after a bee sting"],
       ["that started 10 minutes ago", "beginning a few minutes ago"],
       ["feeling faint", "hoarse voice"],
       ["airway swelling", "wheeze after allergen (anaphylaxis)"], ages=(5, 60)),
    _s("hives_mild", "itchy rash", S,
       ["a few itchy raised welts on the forearms after trying a new soap, with no face swelling or breathing problems",
        "mild hives on the arms after using a new washing powder, breathing normally"],
       ["since this morning", "for a few hours"],
       ["antihistamine is helping"], ages=(10, 60)),
    _s("eczema", "dry itchy skin", N,
       ["dry, itchy, scaly patches on the elbows and behind the knees",
        "persistent itchy red dry skin patches"],
       ["for several months", "for most of the winter"],
       ["moisturiser helps a little", "no fever"], ages=(5, 50)),
    _s("sunburn", "sunburn", S,
       ["red, sore skin on the shoulders after a day at the beach, no blisters",
        "mild sunburn on the back and arms"],
       ["since yesterday", "since the afternoon"],
       ["drinking plenty of water", "no fever"], ages=(10, 60)),

    # ---- injury ----
    _s("ha_injury_anticoag", "head injury", E,
       ["a fall with a blow to the head while taking warfarin, now vomiting repeatedly and confused",
        "hit the head on a step after tripping, is on blood thinners and has since been drowsy and vomiting"],
       ["2 hours ago", "this afternoon"],
       ["headache getting worse"],
       ["head injury on anticoagulants", "vomiting", "confusion"], ages=(65, 90)),
    _s("ha_injury_minor", "head bump", S,
       ["bumped the head on an open cupboard door, leaving a small bump, no loss of consciousness",
        "knocked the forehead on a doorframe, small lump, no vomiting or confusion"],
       ["an hour ago", "this morning"],
       ["feels normal otherwise", "not on blood thinners"], ages=(10, 60)),
    _s("lac_arterial", "bleeding wound", E,
       ["a deep cut to the wrist from broken glass that is spurting blood and not stopping with firm pressure",
        "a deep gash on the thigh with blood pulsing out despite pressing hard on it"],
       ["10 minutes ago", "just now"],
       ["feeling dizzy"],
       ["uncontrolled pulsatile bleeding"]),
    _s("lac_sutures", "cut", U,
       ["a 3 cm gaping cut on the palm from a kitchen knife; bleeding has stopped with pressure",
        "a cut on the finger that is wide open and probably needs stitches, bleeding controlled"],
       ["an hour ago", "this evening"],
       ["can move all fingers normally", "tetanus status unsure"],
       ["wound needing closure"]),
    _s("graze", "graze", S,
       ["a superficial graze on the knee after falling off a bicycle, bleeding stopped",
        "a small scrape on the elbow, cleaned already"],
       ["today", "this afternoon"],
       ["walking normally"], ages=(5, 50)),
    _s("burn_major", "burn", E,
       ["a large scald from boiling water covering the chest and both arms, with blistering and white patches",
        "a big burn over the front of the body after a pan of boiling water tipped over"],
       ["15 minutes ago", "just now"],
       ["screaming in pain"],
       ["large-area burn", "full-thickness appearance"], ages=(1, 10)),
    _s("burn_moderate", "burn", U,
       ["a blistering burn about the size of a palm on the forearm from a hot iron",
        "a burn on the back of the hand that has blistered, roughly 5 cm across"],
       ["2 hours ago", "this morning"],
       ["cooled under running water for 20 minutes"],
       ["partial-thickness burn larger than a coin"]),
    _s("burn_minor", "burn", S,
       ["briefly touched a hot pan, leaving a red sore fingertip with no blistering",
        "a small red mark on the finger from the oven tray, no blister"],
       ["an hour ago", "this evening"],
       ["cooled under the tap already"]),
    _s("ankle_fracture", "ankle injury", U,
       ["a twisted ankle that is very swollen, painful over the bone and cannot take any weight",
        "rolled the ankle playing football, unable to stand on it, bony tenderness at the outer ankle"],
       ["this afternoon", "3 hours ago"],
       ["toes warm and pink"],
       ["unable to weight-bear with bony tenderness (Ottawa rules)"], ages=(12, 60)),
    _s("ankle_sprain", "ankle sprain", S,
       ["a mildly twisted ankle with slight swelling, able to walk with a small limp",
        "a minor ankle sprain, walking on it but a bit sore"],
       ["yesterday", "this morning"],
       ["using ice and rest"], ages=(12, 60)),
    _s("back_cauda", "back pain", E,
       ["lower back pain with new numbness between the legs and difficulty controlling the bladder",
        "back pain and sciatica, now unable to feel the saddle area and has lost bladder control"],
       ["since this morning", "for 1 day"],
       ["both legs feel weak"],
       ["saddle anaesthesia", "bladder dysfunction (cauda equina)"], ages=(25, 70)),
    _s("back_chronic", "back pain", N,
       ["aching lower back pain that has persisted despite stretching, no leg weakness or numbness",
        "ongoing lower back stiffness and ache, no bladder or bowel problems"],
       ["for 7 weeks", "for about 2 months"],
       ["works at a desk all day"], ages=(25, 60)),
    _s("back_strain", "back strain", S,
       ["a mild lower back ache after a day of gardening, already improving",
        "a sore lower back after lifting boxes, getting better with rest and heat"],
       ["for 2 days", "since the weekend"],
       ["no numbness", "no bladder problems"], ages=(20, 60)),

    # ---- ENT / eye / dental ----
    _s("quinsy", "sore throat", U,
       ["a severe one-sided sore throat, unable to swallow saliva, muffled voice and difficulty opening the mouth",
        "the worst sore throat ever, drooling because swallowing hurts too much, voice sounds muffled"],
       ["for 3 days, getting worse", "since the weekend"],
       ["fever", "breathing is fine"],
       ["unable to swallow saliva", "trismus (possible peritonsillar abscess)"], ages=(15, 40)),
    _s("throat_viral", "sore throat", S,
       ["a scratchy sore throat with a cold, still eating and drinking",
        "a mildly sore throat and runny nose"],
       ["for 2 days", "since yesterday"],
       ["no difficulty swallowing", "no rash"], ages=(10, 60)),
    _s("eye_chemical", "eye injury", E,
       ["a splash of oven cleaner in the eye, which is very painful and the vision is blurry",
        "bleach splashed into the eye while cleaning"],
       ["5 minutes ago", "just now"],
       ["eye won't stay open"],
       ["chemical eye injury"]),
    _s("eye_iritis", "painful red eye", U,
       ["a painful red eye with blurred vision and pain when looking at bright lights",
        "one red, aching eye, vision is hazy and light is very uncomfortable"],
       ["since yesterday", "for 1 day"],
       ["no discharge", "wears contact lenses"],
       ["red eye with reduced vision and photophobia"]),
    _s("conjunctivitis", "sticky eye", S,
       ["a red, gritty eye with sticky yellow discharge in the morning; vision is normal and no pain",
        "pink eye with crusting on the lashes, seeing fine"],
       ["for 2 days", "since yesterday"],
       ["no light sensitivity"], ages=(3, 60)),
    _s("dental_abscess", "facial swelling", U,
       ["a toothache with facial swelling spreading up towards the eye and fever",
        "a swollen face from a bad tooth, swelling now reaching the lower eyelid, with a temperature"],
       ["for 2 days", "getting worse since yesterday"],
       ["can open mouth", "breathing normally"],
       ["spreading facial swelling with fever"]),
    _s("toothache", "toothache", N,
       ["a toothache in a back tooth when eating something cold, no swelling",
        "a sensitive, aching molar when chewing"],
       ["for 4 days", "for a week"],
       ["no fever", "no facial swelling"]),
    _s("epistaxis_major", "nosebleed", U,
       ["a nosebleed that has not stopped after 20 minutes of firm pressure, on blood-thinning tablets",
        "a nosebleed still going after 25 minutes of pinching the nose, takes apixaban"],
       ["right now", "for the last 25 minutes"],
       ["no dizziness yet"],
       ["prolonged epistaxis on anticoagulants"], ages=(55, 85)),
    _s("epistaxis_minor", "nosebleed", S,
       ["a nosebleed that stopped after 10 minutes of pinching the soft part of the nose",
        "a short nosebleed after blowing the nose, stopped with pressure"],
       ["this morning", "an hour ago"],
       ["feels fine now"], ages=(8, 50)),
    _s("earache", "earache", S,
       ["mild ear pain with a cold, no discharge, hearing slightly muffled",
        "an ache in one ear alongside a blocked nose"],
       ["for 2 days", "since yesterday"],
       ["no high fever"], ages=(8, 50)),

    # ---- metabolic / cardiac / vascular ----
    _s("dka", "high blood sugar", E,
       ["type 1 diabetes with a blood glucose meter reading 'HI', repeated vomiting, deep rapid breathing and drowsiness",
        "diabetic, very high sugar readings, vomiting, breathing fast and deep, hard to wake"],
       ["since last night", "getting worse over 12 hours"],
       ["fruity smell on breath", "tummy pain"],
       ["hyperglycaemia with vomiting and reduced consciousness (possible DKA)"], ages=(12, 40)),
    _s("hyperglycaemia", "high blood sugar", U,
       ["type 2 diabetes with blood sugar readings consistently above 300 mg/dL, very thirsty but alert and not vomiting",
        "blood glucose staying around 18–20 mmol/L despite taking medication, passing urine a lot"],
       ["for 2 days", "since starting steroid tablets 3 days ago"],
       ["eating and drinking"],
       ["persistent marked hyperglycaemia"], ages=(40, 75)),
    _s("diabetes_review", "diabetes review", N,
       ["wanting a review of diabetes medication because morning readings are slightly higher than target",
        "occasionally a little high fasting sugars and would like the treatment checked"],
       ["for the past month", "over the last few weeks"],
       ["no symptoms", "feeling well"], ages=(40, 75)),
    _s("palp_syncope", "palpitations", E,
       ["a racing, irregular heartbeat with chest pain and fainting",
        "the heart pounding very fast, then passing out briefly"],
       ["20 minutes ago", "this evening"],
       ["still feels faint", "short of breath"],
       ["palpitations with syncope and chest pain"], ages=(30, 80)),
    _s("palp_ectopics", "palpitations", N,
       ["occasional fluttering or skipped beats, mostly after coffee, lasting a second or two",
        "the odd missed heartbeat feeling when resting in the evening"],
       ["for a few weeks", "on and off for a month"],
       ["no fainting", "no chest pain"], ages=(20, 55)),
    _s("dvt", "calf swelling", U,
       ["one calf that is swollen, red and tender after a long-haul flight",
        "the left calf is swollen and painful to walk on, 3 cm bigger than the other one"],
       ["for 2 days", "since returning from a trip"],
       ["no chest pain", "no shortness of breath"],
       ["unilateral calf swelling (possible DVT)"], ages=(25, 75)),
    _s("septic_joint", "hot swollen joint", U,
       ["a single hot, red, very swollen knee that is too painful to move, with a fever",
        "the knee suddenly became swollen and hot, cannot bend it and has a temperature"],
       ["since yesterday", "for 1 day"],
       ["no injury"],
       ["hot swollen joint with fever (possible septic arthritis)"], ages=(20, 80)),
    _s("oa_knees", "joint stiffness", N,
       ["stiff, achy knees, worse after sitting and at the end of the day",
        "gradually worsening knee pain when climbing stairs"],
       ["for several months", "for about a year"],
       ["no swelling or redness", "no fever"], ages=(50, 80)),

    # ---- other ----
    _s("torsion", "testicular pain", E,
       ["sudden severe pain in one testicle with swelling and vomiting",
        "a sudden, very painful testicle that started during sleep, with nausea"],
       ["that started 2 hours ago", "since early this morning"],
       ["no injury"],
       ["acute testicular pain (possible torsion)"], ages=(11, 25), sex="M"),
    _s("overdose_child", "possible poisoning", E,
       ["may have swallowed an unknown number of the grandparent's heart tablets",
        "was found with an open bottle of adult medication and some tablets are missing"],
       ["about 30 minutes ago", "within the last hour"],
       ["seems fine at the moment"],
       ["ingestion of unknown quantity of medication by a child"], ages=(1, 5)),
    _s("suicidal", "low mood", E,
       ["low mood with thoughts of suicide and a specific plan to act on them tonight",
        "hopelessness with a clear intention to end life and access to the means at home"],
       ["for the past week, worse today", "today"],
       ["lives alone"],
       ["suicidal intent with plan"], ages=(18, 70)),
    _s("low_mood", "low mood", N,
       ["low mood, poor sleep and less enjoyment in hobbies, still going to work, no thoughts of self-harm",
        "feeling flat and tired with low motivation, no thoughts of harming themselves"],
       ["for about 6 weeks", "for the past two months"],
       ["supportive family"], ages=(18, 70)),
    _s("ectopic", "pelvic pain in early pregnancy", U,
       ["one-sided lower abdominal pain and light vaginal spotting in early pregnancy (about 6 weeks)",
        "a positive pregnancy test 2 weeks ago and now cramping pain on one side with some spotting"],
       ["since this morning", "for a few hours"],
       ["no fainting", "shoulder feels fine"],
       ["pain and bleeding in early pregnancy (possible ectopic)"], ages=(18, 42), sex="F"),
    _s("preeclampsia", "headache in pregnancy", E,
       ["30 weeks pregnant with a severe headache, flashing lights in the vision and sudden swelling of the face and hands",
        "late pregnancy with a pounding headache, blurred vision and puffy face"],
       ["since this afternoon", "for a few hours"],
       ["pain under the ribs on the right"],
       ["severe headache with visual disturbance in pregnancy (possible pre-eclampsia)"], ages=(18, 42), sex="F"),
    _s("morning_sickness", "nausea in pregnancy", N,
       ["nausea in the mornings in early pregnancy (8 weeks), keeping fluids and some food down",
        "morning sickness at 9 weeks, managing to drink and eat small meals"],
       ["for 2 weeks", "since last week"],
       ["passing urine normally"], ages=(18, 42), sex="F"),
    _s("gastro_child_dehyd", "vomiting and diarrhoea in child", U,
       ["vomiting and diarrhoea with much fewer wet nappies than normal and drinking only small sips",
        "repeated vomiting and watery diarrhoea, only one wet nappy in the last 12 hours"],
       ["for 2 days", "since yesterday"],
       ["more tired than usual but rousable"],
       ["signs of dehydration in a young child"], ages=(1, 4)),
    _s("delirium", "new confusion", U,
       ["new confusion and agitation, more muddled than usual, with smelly urine; breathing and colour normal",
        "suddenly more forgetful and disoriented since yesterday, family suspect a urine infection"],
       ["since yesterday", "over 2 days"],
       ["eating a little", "no fever"],
       ["acute confusion in an older adult"], ages=(75, 95)),
    _s("insect_bite", "insect bite", S,
       ["an itchy red bump from a mosquito bite on the ankle",
        "a small, itchy swollen bite on the arm"],
       ["since yesterday", "for 1 day"],
       ["no spreading redness", "no fever"], ages=(5, 70)),
    _s("cough_persistent", "persistent cough", N,
       ["a dry cough that has lingered after a cold, otherwise feeling well",
        "a tickly cough that just won't go, no blood"],
       ["for 4 weeks", "for about a month"],
       ["no weight loss", "non-smoker"], ages=(20, 70)),
    _s("cough_haemoptysis", "coughing blood", E,
       ["coughing up large amounts of bright red blood, several cupfuls",
        "coughing up lots of fresh blood repeatedly"],
       ["in the last hour", "starting this evening"],
       ["breathless"],
       ["massive haemoptysis"], ages=(35, 80)),
    _s("cough_cold", "cough", S,
       ["a mild cough with a runny nose, otherwise well",
        "a chesty cough with a cold, eating and drinking normally"],
       ["for 3 days", "since the weekend"],
       ["no breathlessness"], ages=(10, 70)),
]

# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
BENIGN_HISTORY = [
    "No significant past medical history.", "Has well-controlled high blood pressure.",
    "Takes no regular medication.", "Has mild hay fever.", "Non-smoker.",
    "Has hypothyroidism on levothyroxine.", "Previously fit and well.", "Has high cholesterol on a statin.",
]
DISTRACTORS = [
    "They mention they have a work deadline this week.", "They recently moved house.",
    "They are worried because a friend had something similar.", "They had a flu vaccine last month.",
    "They are travelling abroad next month.", "They mention their sleep has been fine.",
    "They have been drinking more coffee than usual.", "They ask whether they can still go to the gym.",
]

FIRST_PERSON = {
    "They mention they have a work deadline this week.": "I have a work deadline this week.",
    "They recently moved house.": "I recently moved house.",
    "They are worried because a friend had something similar.": "I'm worried because a friend had something similar.",
    "They had a flu vaccine last month.": "I had a flu vaccine last month.",
    "They are travelling abroad next month.": "I'm travelling abroad next month.",
    "They mention their sleep has been fine.": "My sleep has been fine.",
    "They have been drinking more coffee than usual.": "I've been drinking more coffee than usual.",
    "They ask whether they can still go to the gym.": "Can I still go to the gym?",
}

AGE_WORDS_CHILD = {0: "6-week-old", 1: "1-year-old", 2: "2-year-old", 3: "3-year-old", 4: "4-year-old", 5: "5-year-old"}


def _age_str(age: int) -> str:
    return AGE_WORDS_CHILD.get(age, f"{age}-year-old")


def _render(sc: Scenario, rng: random.Random) -> tuple[str, dict]:
    age = rng.randint(*sc.ages)
    sex = sc.sex if sc.sex != "any" else rng.choice(["F", "M"])
    core = rng.choice(sc.core)
    dur = rng.choice(sc.durations)
    extras = rng.sample(sc.extras, k=min(len(sc.extras), rng.randint(0, 2))) if sc.extras else []
    history = rng.choice(BENIGN_HISTORY) if age >= 16 else ""
    distractor = rng.choice(DISTRACTORS) if rng.random() < 0.4 and age >= 16 else ""

    is_child = age < 12
    if is_child:
        style = "caregiver"
    elif age >= 75:
        style = "clinical"
    else:
        style = rng.choice(["clinical", "patient", "patient"])

    child_noun = ("son" if sex == "M" else "daughter")
    if style == "clinical":
        who = f"{_age_str(age)} {'man' if sex == 'M' else 'woman'}" if age >= 18 else f"{_age_str(age)} {'boy' if sex == 'M' else 'girl'}"
        text = f"{who.capitalize()}. Presenting complaint: {core}; {dur}."
        if extras:
            text += " Associated: " + "; ".join(extras) + "."
        text += " " + history
        if distractor:
            text += " " + distractor
    elif style == "patient":
        text = rng.choice(["I'm {a} ({s}). Main problem: {c} — {d}.",
                           "Hi, {a}{s1} here. My issue: {c}, {d}.",
                           "Age {a}, {s}. Symptoms: {c}; {d}."]).format(
            a=age, s="male" if sex == "M" else "female", s1="M" if sex == "M" else "F", c=core, d=dur)
        if extras:
            text += " Also: " + ", ".join(extras) + "."
        if history:
            text += " Background: " + history
        if distractor:
            text += " " + FIRST_PERSON.get(distractor, "")
    else:  # caregiver
        text = f"About my {_age_str(age)} {child_noun}: {core}, {dur}."
        if extras:
            text += " Also: " + ", ".join(extras) + "."

    meta = {"age": age, "sex": sex, "style": style}
    return " ".join(text.split()), meta


@dataclass
class Case:
    case_id: str
    scenario_id: str
    complaint: str
    text: str
    urgency: str
    red_flags: list[str]
    age: int
    sex: str
    style: str


def generate_cases(cases_per_scenario: int = 8, seed: int = 42) -> list[Case]:
    rng = random.Random(seed)
    cases, seen = [], set()
    for sc in SCENARIOS:
        made, attempts = 0, 0
        while made < cases_per_scenario and attempts < cases_per_scenario * 20:
            attempts += 1
            text, meta = _render(sc, rng)
            if text in seen:
                continue
            seen.add(text)
            cases.append(Case(f"{sc.sid}_{made:02d}", sc.sid, sc.complaint, text, sc.level,
                              list(sc.red_flags), **meta))
            made += 1
    rng.shuffle(cases)
    return cases


def split_cases(cases: list[Case], fractions=(0.2, 0.2, 0.6), grouped: bool = True, seed: int = 42):
    """Split into (train, val, test). If grouped, whole scenarios go to a single split,
    stratified per urgency level so every split contains every level."""
    rng = random.Random(seed)
    if not grouped:
        by_level = {lvl: [c for c in cases if c.urgency == lvl] for lvl in URGENCY_LEVELS}
        out = ([], [], [])
        for lvl_cases in by_level.values():
            rng.shuffle(lvl_cases)
            n = len(lvl_cases)
            a, b = round(n * fractions[0]), round(n * (fractions[0] + fractions[1]))
            out[0].extend(lvl_cases[:a]); out[1].extend(lvl_cases[a:b]); out[2].extend(lvl_cases[b:])
        return tuple(sorted(s, key=lambda c: c.case_id) for s in out)

    assignment = {}
    for lvl in URGENCY_LEVELS:
        sids = sorted({c.scenario_id for c in cases if c.urgency == lvl})
        rng.shuffle(sids)
        n = len(sids)
        n_tr = max(1, round(n * fractions[0]))
        n_va = max(1, round(n * fractions[1]))
        for i, sid in enumerate(sids):
            assignment[sid] = 0 if i < n_tr else (1 if i < n_tr + n_va else 2)
    out = ([], [], [])
    for c in cases:
        out[assignment[c.scenario_id]].append(c)
    return out


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def save_jsonl(cases: list[Case], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(asdict(c)) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_examples(rows: list[dict]) -> list[dspy.Example]:
    """Convert dict rows (case_id, text, urgency, red_flags, ...) into DSPy examples.

    Works for the synthetic data or any external CSV/JSONL with at least
    `text` and `urgency` columns (labels must be one of URGENCY_LEVELS).
    """
    exs = []
    for r in rows:
        ex = dspy.Example(
            symptoms=r["text"],
            urgency_level=r["urgency"],
            red_flags=r.get("red_flags", []),
            case_id=r.get("case_id", ""),
            scenario_id=r.get("scenario_id", ""),
        ).with_inputs("symptoms")
        exs.append(ex)
    return exs
