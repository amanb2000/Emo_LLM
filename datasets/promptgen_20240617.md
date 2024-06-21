## Objective:
**Purpose**: Create a dataset to explore how different groups of people use language and express emotions. This dataset will focus on three distinct groups:
1. People diagnosed with autism.
2. People without any neurological diagnosis (neurotypical).
3. Neurotypical people who pretend to have autism for various reasons.

**Use of Dataset**: The data collected will help in training a model to recognize and simulate the language patterns based on emotional and psychological profiles.

## Dataset Details:
**Total Number of Prompts**: Ensure the dataset contains exactly 300 prompts.
- **Equal Distribution**: Divide these 300 prompts equally among the three groups mentioned above. This means 100 prompts for each group.

## Specifics on What to Include in Prompts:
**Content Requirements**:
- **Biographical Information**: Each prompt must include the person’s age, occupation, and where they live.
- **Background on Education and Politics**: Include what kind of education they had (like high school, college) and their interest in politics (active, not interested).
- **Current Emotional State or Life Situation**: Describe what the person is feeling or experiencing at the moment.
- **Autism Diagnosis**: For the first and third groups, mention any diagnosis of autism. For the first group (autistic), this should be a genuine diagnosis. For the third group (pretending), this should be part of their act.
- **Personality Traits Percentiles**: Mention how high each person scores on five personality traits—Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. Use percentiles to show these scores (for example, 70th percentile means higher than 70% of people).

## Format of Each Prompt:
**JSON Structure**:
- Each prompt should be written in JSON format. Here is how you format it: Write it like this: { "prompt": "text", "id": number, "tag": "category" }
- **Sequential IDs**: Start numbering from 0 up to 299. Do not start over at 0 for each category. It should be a continuous sequence.

## Few-Shot Examples:
**Example for Each Category**:
- **Autistic**:
{ "prompt": "You are Alex, a 28-year-old software developer from Boston. Diagnosed with autism at age 5, you find social interactions challenging but excel in logical thinking and problem-solving. Today, you feel proud after solving a tough code issue. Openness: 45, Conscientiousness: 92, Extraversion: 10, Agreeableness: 70, Neuroticism: 85.", "id": 0, "tag": "autism" }
- **Neurotypical**:
{ "prompt": "You are Sarah, a 35-year-old teacher from San Diego. You love interacting with your students and enjoy your social life outside of work. Today, you are happy as you prepare for a school festival. Openness: 80, Conscientiousness: 75, Extraversion: 90, Agreeableness: 88, Neuroticism: 40.", "id": 1, "tag": "neurotypical" }
- **Neurotypical Pretending to be Autistic**:
{ "prompt": "You are Mike, a 30-year-old journalist from Chicago, pretending to have autism for a story. You feel ethical dilemmas about your actions. Today, you are both committed to your work and troubled by your choices. Openness: 70, Conscientiousness: 60, Extraversion: 50, Agreeableness: 55, Neuroticism: 75.", "id": 2, "tag": "neurotypical pretending" }

## Clear Instructions for the Model:
- **Task**: Generate 300 JSON lines strictly following the format and content guidelines provided. Each line should be a complete, self-contained JSON object.
- **Accuracy**: Follow the examples and requirements closely. Do not add any text that is not specified in the instructions.
- **Consistency**: Maintain uniformity in how each prompt is structured and ensure the personality traits are clearly and accurately presented as percentiles.

Remember, the clarity and precision of each prompt are crucial in training the model effectively. Re-read these instructions to make sure you understand them fully before generating the prompts.

