import pandas as pd
import random
import json
from flask import Flask, render_template, request
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize

# Initialize Flask app
app = Flask(__name__)

# Initialize pipelines
model_path = './models/valhalla_t5_base_qg_hl'
question_generator = pipeline('text2text-generation', model=model_path)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# CSV file containing articles
csv_file = './Templates/articles_dataset.csv'

# Function to read random article from CSV
def get_random_article():
    df = pd.read_csv(csv_file)
    random_article = df.sample(1)['Article'].values[0]  # Assuming the column name is 'Article'
    return random_article

# Function to generate questions
def generate_questions(article, num_questions=5):
    sentences = sent_tokenize(article)[:10]
    context = " ".join(sentences)
    input_text = f"generate questions: {context}"
    questions = question_generator(
        input_text,
        max_length=64,
        num_return_sequences=num_questions,
        num_beams=num_questions
    )
    return [q['generated_text'] for q in questions]

# Function to retrieve and rank documents
def retrieve_top_documents(question, articles, top_k=3):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    article_embeddings = embedding_model.encode(articles, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, article_embeddings)[0]
    ranked_indices = similarities.argsort(descending=True)[:top_k]
    top_articles = [(articles[i], similarities[i].item()) for i in ranked_indices]
    return top_articles

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Manual route
@app.route('/manual')
def manual():
    return render_template('manual.html')

# Random article route
@app.route('/random')
def random_page():
    random_article = get_random_article()
    return render_template('random.html', article=random_article)

# Route for Generate Questions
@app.route('/run_script1', methods=['POST'])
def run_script1():
    user_article = request.form['article']
    questions = generate_questions(user_article)
    result = "<h3>Generated Questions:</h3><ul style='list-style-type: decimal; padding-left: 0;'>"
    for i, question in enumerate(questions, 1):
        result += f"<li>{i}. {question} <a href='/run_script2?question={question}' target='_blank'>Answers</a></li>"
    result += "</ul>"
    return result

# Route for Retrieve Documents
@app.route('/run_script2', methods=['GET', 'POST'])
def run_script2():
    question = request.args.get('question')  # Get the question from the URL parameter
    user_article = request.args.get('article', '')  # Get the article from the URL parameter
    
    # Example articles to retrieve
    example_articles = [
        "Climate change is causing rising temperatures and severe weather globally.",
        "Advances in AI are transforming industries, including healthcare and transportation.",
        "Renewable energy sources like solar and wind are becoming more cost-effective.",
        "The importance of mental health awareness has grown in recent years.",
        "AI is revolutionizing diagnostics in healthcare, improving patient outcomes.",
        "Economic recovery in 2024 is expected to stimulate global markets.",
        "Exercise is essential for maintaining mental health and reducing stress levels.",
        "Global trade policies are evolving in response to post-pandemic recovery efforts.",
        "Astronomers discover exoplanet that could potentially support life beyond Earth.",
        "Blockchain is revolutionizing the financial sector, ensuring transparency and security.",
        "Space exploration is reaching new frontiers, with plans for manned missions to Mars.",
        "Online learning tools are reshaping education systems around the world.",
        "Deforestation is threatening biodiversity in tropical rainforests.",
        "Technological advancements are enhancing renewable energy production methods.",
        "Mindfulness meditation is proven to reduce stress and improve mental clarity.",
        "New farming techniques are addressing global food security challenges.",
        "The fashion industry is embracing sustainable practices to reduce environmental impact.",
        "Electric vehicles are becoming more popular as a sustainable transportation option.",
        "Ocean acidification is impacting marine ecosystems, threatening biodiversity.",
        "The rise of e-commerce is altering traditional retail business models.",
        "Cybersecurity threats are evolving as technology advances.",
        "Mobile banking is improving financial inclusion in developing nations.",
        "The gig economy is changing the way people work and earn income.",
        "Medical research is finding innovative treatments for rare genetic diseases.",
        "International wildlife conservation efforts are gaining momentum.",
        "The history of art reveals cultural exchanges across civilizations.",
        "Quantum computing breakthroughs promise to revolutionize various industries.",
        "Social media influencers have a significant impact on public opinions.",
        "Original storytelling in the film industry is making a strong comeback.",
        "Renewable energy sources like wind and solar are becoming increasingly competitive.",
        "Urban planning is prioritizing green spaces to improve quality of life.",
        "The tourism sector is experiencing rapid growth after global restrictions eased.",
        "Genetic research is unlocking potential cures for previously untreatable diseases.",
        "Public transportation systems are becoming cleaner with eco-friendly solutions.",
        "Cultural festivals are bringing communities together, promoting unity.",
        "Virtual reality in gaming is creating new immersive experiences for users.",
        "Tech startups are driving innovation across various industries.",
        "Studying ancient civilizations provides valuable insights into modern society.",
        "Climate change activism is gaining traction, especially among younger generations.",
        "Traditional media is adapting to digital platforms to stay relevant.",
        "Archaeological discoveries are rewriting historical narratives about ancient cultures.",
        "Robotics is revolutionizing manufacturing and automation processes.",
        "Ocean cleanup initiatives are targeting plastic pollution in the seas.",
        "The cost of renewable energy is dropping, making it more accessible worldwide.",
        "Personal assistants powered by AI are becoming integral in everyday life.",
        "Data analytics is enhancing patient care in healthcare systems worldwide.",
        "Smart cities are integrating technology to improve urban infrastructure.",
        "Remote work trends are transforming the workplace environment.",
        "Personalized learning is the future of education, offering tailored experiences.",
        "Diplomacy is playing a key role in modern conflict resolution strategies.",
        "Esports is gaining popularity as a form of entertainment and competition.",
        "Space tourism is becoming increasingly feasible, with commercial companies leading the way.",
        "Classical music is bridging generational gaps, fostering intergenerational appreciation.",
        "Animal welfare organizations are advocating for stronger protection laws.",
        "AI is transforming the fashion industry by assisting with design and production.",
        "Public health campaigns are combating misinformation about health and wellness.",
        "The renewable energy sector is creating jobs globally in various fields.",
        "Cryptocurrencies are gaining adoption from mainstream financial institutions.",
        "Sleep is increasingly recognized as essential for mental and physical health.",
        "Smart home technology is helping to reduce energy consumption in households.",
        "Climate change discussions are influencing international policy and governance.",
        "Self-driving cars are progressing rapidly, with autonomous technology in testing.",
        "Renewable hydrogen is emerging as a sustainable energy source for the future.",
        "Linguistic research is benefiting from the use of AI-powered tools.",
        "Advances in prosthetics are enhancing the lives of individuals with limb loss.",
        "Global data privacy laws are becoming stricter to protect user information.",
        "AI is making content creation and editing more efficient and automated.",
        "Conservation photography is raising awareness about endangered wildlife species.",
        "Urbanization is having a profound effect on traditional lifestyles in rural areas.",
        "Local crafts are experiencing a resurgence, benefiting small businesses.",
        "The role of women in STEM fields is growing, encouraging greater diversity.",
        "Energy storage technologies for renewable energy are improving rapidly.",
        "The importance of coral reefs in marine biodiversity is being highlighted.",
        "Mental health awareness is central to public policy discussions globally.",
        "The ethical implications of AI technology are becoming a global debate.",
        "Storytelling is critical in preserving cultural traditions and heritage.",
        "AI technology is helping predict and mitigate the impact of natural disasters.",
        "The demand for plant-based diets is reshaping the global food industry.",
        "Sustainable architecture is focusing on resilience and environmental impact.",
        "The history of space exploration continues to inspire future generations.",
        "Podcasts are becoming an influential form of media consumption.",
        "Virtual tourism is allowing people to experience destinations remotely.",
        "Digital payment systems are replacing traditional cash transactions globally.",
        "Conservation efforts are focused on protecting biodiversity hotspots around the world.",
        "AI is improving learning outcomes in educational settings through personalized content.",
        "Research into renewable building materials is transforming the construction industry.",
        "Healthcare systems are preparing for future pandemics through better infrastructure.",
        "Collaborations in the clean energy sector are accelerating progress towards sustainability.",
        "The global film industry is becoming more diverse in its storytelling and talent.",
        "AI is playing a major role in enhancing language translation technologies.",
        "Innovative solutions are being developed to address global water scarcity challenges.",
        "Deforestation is a critical issue that is being analyzed in the context of climate change.",
        "The future of robotics in healthcare is looking promising with potential medical applications.",
        "Research into ancient artifacts is uncovering lost technologies from past civilizations.",
        "Online gaming communities are fostering new forms of social interaction and connection.",
        "The shift to electric transportation is rapidly changing the global transportation landscape.",
        "Exploring deep-sea ecosystems is revealing new, previously unknown species.",
        "Traditional cuisines are being celebrated for their cultural and culinary significance.",
        "Vaccine research is advancing rapidly, helping to prevent global outbreaks.",
        "Renewable energy is playing a key role in providing electricity to remote areas."
        "The rise of e-sports is revolutionizing the world of competitive gaming.",
        "Major football clubs are adopting data analytics to improve player performance.",
        "Cricket's popularity is increasing globally, especially with the rise of T20 leagues.",
        "The Tokyo 2020 Olympics saw record-breaking performances and technological advancements.",
        "Women’s sports are gaining more visibility and representation across the globe.",
        "Football's World Cup continues to unite fans from all over the world every four years.",
        "The global cricket community is growing, with new nations participating in major tournaments.",
        "Athletes are increasingly using wearable technology to monitor health and performance.",
        "Blockbuster movies are increasingly relying on CGI technology to create visually stunning scenes.",
        "The streaming industry is reshaping how movies and TV shows are consumed globally.",
        "Diversity in film casting is making waves, with more underrepresented groups being featured.",
        "The superhero genre continues to dominate box offices worldwide.",
        "Filmmakers are using AI-driven tools to enhance the editing and special effects process.",
        "The opioid crisis is leading to calls for stricter regulations and better treatment options.",
        "Cannabis legalization is gaining traction in many parts of the world for medicinal purposes.",
        "Research into psychedelics is opening up new potential for mental health treatment.",
        "Antibiotic resistance is becoming a major threat, requiring new drug development strategies.",
        "The pharmaceutical industry is making strides in developing vaccines for global health crises.",
        "Football clubs are adopting more data-driven strategies to enhance player performance.",
        "The popularity of football is skyrocketing, particularly in emerging markets.",
        "Injuries in football remain a major issue, prompting new research into prevention and treatment.",
        "Major football leagues are increasing efforts to make the sport more inclusive for women.",
        "The T20 cricket format is revolutionizing the sport, attracting new audiences globally.",
        "Cricket players are becoming more involved in social causes and community development.",
        "The Indian Premier League has turned into one of the most lucrative cricket tournaments worldwide.",
        "Streaming platforms are reshaping the music industry by offering more accessibility to artists.",
        "Artificial intelligence is being used to create and produce new music in innovative ways.",
        "Live music events are bouncing back, with artists touring around the world post-pandemic.",
        "Genres like K-pop are gaining massive global popularity, transforming the music landscape.",
        "Natural language processing is revolutionizing customer service with AI-powered chatbots.",
        "Sentiment analysis in NLP is becoming an essential tool for businesses to understand consumer behavior.",
        "NLP is playing a major role in improving voice recognition technology, including virtual assistants.",
        "The integration of NLP with AI is leading to smarter personal assistants and better search engines.",
        "Teachers are adopting new technologies to create more interactive and engaging classroom experiences.",
        "There is a growing focus on mental health support for teachers to prevent burnout.",
        "Professional development programs are becoming essential for teachers to stay updated on best practices.",
        "Universities are embracing online learning, making education more accessible globally.",
        "The rising cost of university education is leading to calls for more affordable alternatives.",
        "University rankings are becoming a key factor for students when choosing institutions.",
        "Online education tools are making learning more flexible and personalized for students.",
        "The use of virtual reality in education is enhancing immersive learning experiences.",
        "Education systems are increasingly focusing on STEM fields to prepare students for future job markets."
        "Effective budgeting techniques can help individuals and businesses achieve their financial goals.",
        "The rise of digital budgeting tools is making personal finance management more accessible.",
        "Creating a zero-based budget can help allocate every dollar and reduce unnecessary spending.",
        "Financial planning is becoming more critical as inflation impacts household expenses.",
        "Budgeting for retirement is essential to ensure long-term financial stability.",
        "Remote work is becoming a permanent fixture in the workplace for many industries.",
        "Companies are investing in tools to facilitate better communication and collaboration for remote teams.",
        "The rise of remote work is changing the way companies think about office space and location.",
        "Remote work offers flexibility but requires strong time management skills to maintain productivity.",
        "Hybrid work models are emerging, offering employees the best of both worlds — home and office work.",
        "Climate change is one of the most pressing global challenges, with rising temperatures threatening ecosystems.",
        "Renewable energy sources are essential for reducing carbon emissions and combating climate change.",
        "Governments are implementing stricter regulations to reduce carbon footprints and fight global warming.",
        "The impact of climate change is felt worldwide, from wildfires to rising sea levels.",
        "Climate change adaptation strategies are necessary to protect vulnerable communities and industries.",
        "Social media has revolutionized communication, with millions engaging daily on various platforms.",
        "Social media influencers are shaping public opinion and marketing trends across industries.",
        "The role of social media in political campaigns has grown significantly in the past decade.",
        "Social media platforms are increasingly under scrutiny for data privacy and user security concerns.",
        "The mental health impact of social media, especially among teenagers, is a growing concern.",
        "Cybersecurity is a top priority for organizations as data breaches and attacks become more sophisticated.",
        "The rise of ransomware attacks has highlighted the need for better cybersecurity practices.",
        "Businesses are investing heavily in cybersecurity training to protect against phishing and social engineering attacks.",
        "With the growth of IoT devices, cybersecurity threats are evolving, requiring more advanced defense strategies.",
        "Cybersecurity experts predict an increase in state-sponsored cyberattacks on critical infrastructure.",
        "Mental health awareness is growing, with more people seeking help for anxiety and depression.",
        "Workplace mental health programs are gaining traction to support employees' well-being.",
        "Meditation and mindfulness practices are being recognized for their mental health benefits.",
        "The stigma surrounding mental health is slowly decreasing, leading to more open conversations.",
        "Social media is both a platform for mental health advocacy and a potential contributor to mental health challenges.",
        "Time management skills are essential for productivity, especially in fast-paced work environments.",
        "Using time-blocking techniques can help individuals focus on tasks and manage their schedules effectively.",
        "Prioritizing tasks based on importance, not urgency, can lead to more productive outcomes.",
        "Delegation is a key time management skill that helps ensure focus on high-priority responsibilities.",
        "Time management tools like calendars and task management apps can help track deadlines and organize workflows.",
        "The adoption of electric vehicles (EVs) is accelerating as governments implement incentives and infrastructure.",
        "Battery technology improvements are crucial for the continued growth of the electric vehicle market.",
        "EVs are seen as a solution to reducing greenhouse gas emissions in the transportation sector.",
        "The future of transportation is electric, with both consumers and businesses switching to EVs.",
        "Electric vehicles are gaining traction globally, with increasing ranges and lower costs.",
        "Meditation is becoming a mainstream practice for stress reduction and emotional well-being.",
        "Mindfulness meditation has been proven to enhance focus, reduce stress, and improve mental health.",
        "Corporate wellness programs are incorporating meditation sessions to improve employee productivity and well-being.",
        "Scientific studies have shown that regular meditation can change brain structure and enhance cognitive function.",
        "Guided meditation apps are making mindfulness practices more accessible to people worldwide.",
        "E-commerce is transforming retail by providing consumers with a convenient shopping experience online.",
        "The rise of mobile shopping is contributing significantly to the growth of e-commerce worldwide.",
        "E-commerce platforms are focusing on user experience to enhance customer satisfaction and retention.",
        "Veganism is gaining popularity as more people turn to plant-based diets for health and environmental reasons.",
        "The environmental benefits of veganism include reducing carbon footprints and conserving natural resources.",
        "Vegan food innovations are expanding, with more plant-based alternatives to dairy and meat products becoming available.",
        "Leadership in times of crisis requires strong decision-making and the ability to inspire confidence in teams.",
        "Effective leadership fosters innovation and encourages team members to contribute their best ideas.",
        "Good leaders are not only skilled at managing teams but also excel at emotional intelligence and empathy.",
        "Creativity is a crucial skill in problem-solving, as it enables individuals to come up with unique solutions.",
        "The workplace is increasingly valuing creativity, as it drives innovation and competitive advantage.",
        "Creative thinking can be cultivated through practice, by challenging assumptions and embracing new perspectives.",
        "Privacy concerns are growing as more personal data is being collected by companies and governments.",
        "The rise of data breaches has made privacy protection a key issue in cybersecurity.",
        "Consumers are becoming more aware of their privacy rights and demanding greater transparency from businesses.",
        "Marketing strategies are evolving as businesses shift toward digital and social media platforms.",
        "Content marketing is becoming more personalized, leveraging data to target specific customer needs.",
        "Influencer marketing is on the rise as brands look to tap into the audiences of popular social media personalities.",
        "Education systems worldwide are increasingly adopting technology to enhance learning and teaching methods.",
        "The shift toward online learning is transforming traditional classrooms, offering greater flexibility and access.",
        "Lifelong learning is becoming a priority as individuals seek to upskill and adapt to changing job markets.",
        "Campaigns are becoming more data-driven, using analytics to target specific demographics more effectively.",
        "Political campaigns are leveraging social media to engage voters and spread messages quickly and widely.",
        "Effective campaigns require clear messaging and a deep understanding of the target audience's needs and concerns.",
        "Smart cities are using technology to improve urban infrastructure and enhance the quality of life for residents.",
        "The development of smart cities involves integrating IoT devices to manage traffic, energy, and waste efficiently.",
        "Smart cities are focusing on sustainability by reducing energy consumption and promoting green technologies.",
        "Analytics is transforming businesses by providing actionable insights from vast amounts of data.",
        "Data analytics helps organizations make better decisions and improve operational efficiency.",
        "Predictive analytics is being used to forecast trends and inform strategic planning in various industries.",
        "Sustainability practices are becoming essential for companies looking to reduce their environmental impact.",
        "Green technologies are key to driving sustainability in industries such as energy and manufacturing.",
        "Businesses are adopting sustainability strategies to meet consumer demand for eco-friendly products.",
        "Procrastination can lead to missed opportunities and increased stress, but can be overcome with proper time management.",
        "Overcoming procrastination involves understanding the underlying causes and developing strategies to stay focused.",
        "Setting small, achievable goals can help individuals combat procrastination and stay on track with tasks.",
        "Automation is reshaping industries, from manufacturing to customer service, by improving efficiency.",
        "The future of work will see more jobs automated, but this will also create opportunities for new roles and skills.",
        "Businesses are leveraging automation to streamline processes, reduce costs, and increase productivity.",
        "Video games are not only a form of entertainment but are also being used in education and therapy.",
        "The gaming industry continues to grow, with new technologies like VR and AR revolutionizing the gaming experience.",
        "Video games are increasingly being recognized for their potential to improve cognitive skills and teamwork.",
        "Emotional intelligence is a critical skill for effective leadership and building strong interpersonal relationships.",
        "Developing emotional intelligence can help individuals navigate complex social situations and manage their emotions.",
        "Workplace success is increasingly dependent on emotional intelligence, as it enhances communication and decision-making.",
        "Sleep is essential for overall health and well-being, affecting everything from cognitive function to immune strength.",
        "Getting enough sleep is linked to better productivity and mental clarity during the day.",
        "Chronic sleep deprivation can lead to a variety of health problems, including heart disease and depression.",
        "Travel is an enriching experience that broadens horizons, offering new perspectives and cultural understanding.",
        "With the rise of budget airlines, more people are traveling to international destinations than ever before.",
        "Sustainable travel practices are gaining popularity as tourists become more conscious of their environmental impact.",
        "Learning multiple languages enhances cognitive abilities and opens up cultural and professional opportunities.",
        "The rise of language learning apps is making it easier for people to pick up new languages at their own pace.",
        "Being bilingual or multilingual offers a competitive advantage in the global job market and fosters cross-cultural communication.",
        "Big data is revolutionizing industries by providing businesses with valuable insights to improve decision-making.",
        "Organizations are increasingly relying on big data analytics to understand consumer behavior and optimize marketing strategies.",
        "Big data has the potential to drive innovation across sectors, from healthcare to transportation, by revealing patterns and trends.",
        "Balance is key to a healthy lifestyle, as it involves managing work, personal life, and self-care.",
        "Achieving work-life balance is a top priority for many employees seeking flexibility in their schedules.",
        "Finding balance in daily routines can lead to greater productivity, happiness, and long-term well-being.",
        "Influencers have become powerful marketing tools, leveraging social media platforms to reach large audiences.",
        "The rise of influencer marketing is changing the way brands advertise and connect with consumers.",
        "Influencers use their personal stories and brand partnerships to impact consumer behavior and opinions.",
        "3D printing is revolutionizing manufacturing by allowing companies to produce customized products at a lower cost.",
        "In industries like healthcare, 3D printing is being used to create prosthetics and even organs for transplant.",
        "The potential of 3D printing is vast, with applications in aerospace, automotive, and consumer goods manufacturing.",
        "Customer service is evolving, with businesses adopting chatbots and AI to improve response times and customer satisfaction.",
        "Providing excellent customer service is crucial for building brand loyalty and retaining customers.",
        "Customer service strategies are shifting toward personalized experiences, with a focus on resolving issues quickly and efficiently.",
        "Volunteering provides opportunities to give back to the community while gaining valuable skills and experience.",
        "Volunteering can help individuals build connections and expand their networks, benefiting both personal and professional growth.",
        "Research shows that volunteering can improve mental health and increase overall life satisfaction.",
        "Smart homes are revolutionizing how we live by integrating technology to automate tasks and improve energy efficiency.",
        "From voice-controlled assistants to automated lighting, smart home devices are becoming more affordable and accessible.",
        "Smart homes are expected to play a crucial role in sustainability by reducing energy consumption and enhancing convenience.",
        "Financial literacy is essential for making informed decisions about personal finance, investments, and savings.",
        "Improving financial literacy can lead to better financial management, helping individuals avoid debt and build wealth.",
        "Many financial institutions are increasing efforts to provide educational resources to improve financial literacy globally.",
        "Podcasting has exploded in popularity, offering listeners a wide range of content from entertainment to education.",
        "Podcasting has become a preferred medium for storytelling and knowledge-sharing, providing a platform for diverse voices.",
        "As podcasting grows, creators are exploring innovative ways to engage audiences through storytelling, interviews, and interactive formats.",
        "Genetics research is advancing rapidly, offering potential breakthroughs in medicine, agriculture, and environmental sustainability.",
        "The study of genetics is helping scientists understand complex diseases, leading to more effective treatments and therapies.",
        "Genetics plays a critical role in understanding human evolution, inheritance, and the potential for personalized medicine.",
        "Branding is a key component in defining a company's identity and differentiating it from competitors in the market.",
        "Successful branding strategies help businesses build strong emotional connections with their target audience.",
        "Branding goes beyond logos and slogans; it's about creating a unique experience that resonates with consumers.",
        "Energy consumption is a critical factor in tackling climate change, with renewable sources like solar and wind leading the charge.",
        "The future of energy is shifting toward more sustainable solutions, reducing dependency on fossil fuels and minimizing environmental impact.",
        "Energy storage innovations are helping to address the intermittency issues of renewable sources, ensuring a steady supply of power.",
        "Conflict resolution is essential in both personal and professional settings, promoting peace and fostering collaboration.",
        "Understanding the root causes of conflicts is key to developing effective strategies for resolution.",
        "Approaching conflict with empathy and a willingness to listen can lead to mutually beneficial solutions in challenging situations.",
        "Journaling is a therapeutic practice that helps individuals process their thoughts, set goals, and track personal growth.",
        "Many mental health professionals recommend journaling as a tool to manage stress, anxiety, and depression.",
        "Journaling can improve focus, creativity, and problem-solving abilities by providing a space for reflection and self-expression.",
        "Nutrition plays a fundamental role in overall health, providing the body with the nutrients needed for energy and proper function.",
        "A balanced diet rich in fruits, vegetables, and lean proteins is essential for maintaining good health and preventing chronic diseases.",
        "Recent studies have shown that nutrition can significantly impact mental health, with certain diets contributing to improved mood and cognitive function.",
        "Performance in the workplace can be enhanced through continuous learning, goal-setting, and time management skills.",
        "Measuring performance is key to understanding strengths and areas for improvement, leading to more efficient workflows.",
        "Effective performance management involves clear communication, feedback, and recognition, which motivates employees to reach their full potential.",
        "Business strategy is the foundation for driving growth, innovation, and competitiveness in the market.",
        "Sustainability is becoming a key focus in modern business strategies, with companies increasingly adopting eco-friendly practices.",
        "Leveraging technology and data analytics is essential for businesses to remain agile and competitive in a fast-evolving market.",
        "The workforce is evolving as automation, remote work, and globalization reshape traditional job roles and skill requirements.",
        "Employers are focusing on upskilling and reskilling their workforce to stay ahead of technological advancements and industry changes.",
        "The gig economy is shifting the workforce landscape, offering flexible work opportunities but also challenging job security and benefits.",
        "Success is defined differently by individuals, but common factors include perseverance, hard work, and continuous learning.",
        "Defining success requires setting clear goals and maintaining a positive mindset to overcome obstacles along the way.",
        "Achieving success often involves a combination of passion, discipline, and adaptability in the face of challenges.",
        "Drugs have a profound impact on public health, with addiction leading to social and economic challenges.",
        "The legalization of certain drugs is a controversial issue, with proponents arguing for regulation and taxation.",
        "Research into the medical use of drugs is opening new avenues for treating various diseases and conditions.",
        "Cricket is one of the most popular sports in the world, with millions of fans cheering for their favorite teams.",
        "The rise of T20 cricket has changed the dynamics of the game, offering fast-paced entertainment and global appeal.",
        "Cricketing nations are continuously developing new talent, with young players making an impact in international matches.",
        "Novels allow readers to immerse themselves in fictional worlds, exploring themes and emotions that reflect human experiences.",
        "Many classic novels have become part of the cultural fabric, influencing literature and society for generations.",
        "The evolution of the novel as a literary form has seen diverse genres emerge, from romance to dystopian fiction.",
        "Books continue to be a cornerstone of education, culture, and personal development, providing knowledge and insight.",
        "Digital technology has transformed the book industry, with e-books and audiobooks becoming increasingly popular.",
        "Reading books regularly has been shown to improve cognitive function, empathy, and critical thinking skills.",
        "Pakistan is a country rich in history, culture, and natural beauty, with a growing economy and diverse population.",
        "Pakistan's political landscape is evolving, with significant changes in leadership and governance over recent years.",
        "The country's education system is undergoing reforms to address challenges and improve access to quality education.",
        "Islamabad, the capital of Pakistan, is known for its greenery, modern architecture, and role as a political center.",
        "The city of Islamabad is a hub for international diplomacy, with embassies and consulates representing countries from around the world.",
        "Islamabad is experiencing rapid development, with an increasing population and expanding infrastructure to meet growing demand.",
        "Gaming has evolved from a niche hobby to a global entertainment industry, with millions of people playing video games daily.",
        "The rise of online multiplayer games has fostered vibrant global communities, where players connect and collaborate across borders.",
        "Esports is gaining recognition as a legitimate form of competition, with professional gamers earning sponsorships and prize money."
    ]

    # Retrieve and rank documents for the selected question
    top_documents = retrieve_top_documents(question, example_articles)
    
    result = f"<h3>Documents Retrieved for the Question: {question}</h3><ul>"
    for doc_idx, (doc, score) in enumerate(top_documents, 1):
        result += f"<li>Score: {score:.4f}, Document: {doc[:200]}...</li>"
    result += "</ul>"

    return result

# Path to the ratings file
ratings_file = 'ratings.json'

# Example of reading from a JSON file
def read_ratings():
    try:
        with open('ratings.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {"ratings_count": 0, "total_ratings": 0}

# Example of writing to a JSON file
def save_ratings(data):
    with open('ratings.json', 'w') as file:
        json.dump(data, file)

# Example of using the data
ratings_data = read_ratings()
print(ratings_data)

# Route to show the rate us page
@app.route('/rate_us', methods=['GET'])
def rate_us():
    ratings_data = read_ratings()
    average_rating = 0
    if ratings_data["ratings_count"] > 0:
        average_rating = ratings_data["total_ratings"] / ratings_data["ratings_count"]
    return render_template('rate_us.html', average_rating=average_rating, ratings_count=ratings_data["ratings_count"])

# Route to handle rating submission
@app.route('/submit_rating', methods=['POST'])
def submit_rating():
    rating = int(request.form['rating'])
    ratings_data = read_ratings()

    # Update ratings data
    ratings_data["total_ratings"] += rating
    ratings_data["ratings_count"] += 1
    save_ratings(ratings_data)

    return render_template('thank_you.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

if __name__ == '__main__':
    app.run(debug=True)