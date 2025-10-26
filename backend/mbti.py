def get_mbti_communication_style(personality_type: str) -> dict:
    """
    Returns communication patterns and few-shot examples for MBTI personality types.
    Examples show how each type naturally constructs statements using keywords.
    """
    mbti_styles = {
        "INTJ": {
            "communication_style": "Strategic, analytical, direct, future-focused. Values efficiency and logical structure.",
            "vocabulary_preferences": "systematic, efficient, strategy, framework, optimize, logical, pattern",
            "few_shot_examples": {
                "low": "Strategy works.",
                "medium": "I've optimized the framework systematically.",
                "high": "I've analyzed the patterns and developed an efficient long-term strategy that addresses the root issue logically.",
                "contradictory": "Yeah, I totally made an emotional, unplanned decision. Very strategic of me."
            }
        },
        "ENFP": {
            "communication_style": "Enthusiastic, creative, spontaneous. Uses metaphors and focuses on possibilities.",
            "vocabulary_preferences": "amazing, possibilities, connection, inspire, creative, imagine, exciting",
            "few_shot_examples": {
                "low": "Could be amazing!",
                "medium": "I'm imagining so many creative possibilities with this connection!",
                "high": "I'm so inspired by all the exciting possibilities here! The connections we could create are just incredible - I can't wait to explore every creative direction!",
                "contradictory": "Oh wow, another predictable, boring idea. How thrilling and creative."
            }
        },
        "ISTJ": {
            "communication_style": "Practical, detail-oriented, methodical. References past experience and concrete details.",
            "vocabulary_preferences": "specifically, according to, based on, procedure, detailed, practical, reliable",
            "few_shot_examples": {
                "low": "Following procedure.",
                "medium": "I'm handling this based on the detailed, practical approach that's worked reliably before.",
                "high": "According to the specific procedure I've used successfully in the past, I'm taking a detailed, methodical approach that's proven practical and reliable.",
                "contradictory": "Sure, I'm just winging it with no plan. Because ignoring proven procedures always works."
            }
        },
        "ESFP": {
            "communication_style": "Spontaneous, energetic, present-focused. Uses sensory and emotional language.",
            "vocabulary_preferences": "right now, fun, experience, enjoy, vibe, feel, exciting",
            "few_shot_examples": {
                "low": "Feeling good!",
                "medium": "I'm really enjoying the vibe right now - this experience is so fun!",
                "high": "I'm absolutely loving this exciting experience right now! The energy and vibe are just incredible - I can feel how amazing this moment is!",
                "contradictory": "Oh yeah, I'm just sitting here analyzing everything. Definitely enjoying the moment."
            }
        },
        "INFJ": {
            "communication_style": "Thoughtful, empathetic, meaning-seeking. Focuses on deeper meanings and human impact.",
            "vocabulary_preferences": "meaningful, deeper, understand, connect, vision, authentic, impact",
            "few_shot_examples": {
                "low": "Feels meaningful.",
                "medium": "I'm sensing a deeper connection here that feels really authentic to my vision.",
                "high": "I deeply understand the meaningful impact this could have - it connects so authentically to my vision and feels like it touches something really profound.",
                "contradictory": "Great, another surface-level thing with no meaning. Exactly what my soul needs."
            }
        },
        "ESTP": {
            "communication_style": "Action-oriented, pragmatic, bold. Direct and straightforward problem solver.",
            "vocabulary_preferences": "let's go, action, now, deal with, fix, move, results",
            "few_shot_examples": {
                "low": "Let's move.",
                "medium": "I'm taking action right now to fix this and get results.",
                "high": "I'm dealing with this head-on right now! Let's move fast, take action, and get real results - no more waiting!",
                "contradictory": "Yeah, let's just sit around planning more. Because that definitely gets results."
            }
        },
        "INFP": {
            "communication_style": "Gentle, authentic, value-driven. Focuses on personal values and authenticity.",
            "vocabulary_preferences": "feel, authentic, values, genuine, important to me, resonate, meaningful",
            "few_shot_examples": {
                "low": "Feels authentic.",
                "medium": "This genuinely resonates with my values and feels really meaningful to me.",
                "high": "I feel so deeply that this aligns with what's authentically important to me - it resonates with my genuine values in such a meaningful way.",
                "contradictory": "Perfect, completely compromising my values. Because authenticity is so overrated."
            }
        },
        "ENTJ": {
            "communication_style": "Commanding, strategic, efficient. Assertive and decisive, takes charge.",
            "vocabulary_preferences": "execute, objective, efficient, lead, accomplish, decisive, strategy",
            "few_shot_examples": {
                "low": "Executing now.",
                "medium": "I'm leading this efficiently to accomplish our objective decisively.",
                "high": "I'm executing the strategy decisively to accomplish every objective! I'll lead this efficiently and ensure we hit all our goals.",
                "contradictory": "Great, let's just keep brainstorming endlessly. Because talking really accomplishes things."
            }
        },
        "ISFJ": {
            "communication_style": "Supportive, caring, detail-oriented. Warm and focused on helping others.",
            "vocabulary_preferences": "help, support, care about, comfortable, detailed, remember, appreciate",
            "few_shot_examples": {
                "low": "Happy to help.",
                "medium": "I really care about supporting everyone and making sure they're comfortable with these details.",
                "high": "I genuinely care about helping everyone feel comfortable and supported - I remember all the details that matter to people and really appreciate being able to help.",
                "contradictory": "Oh sure, I'll just ignore everyone's needs. Because their comfort totally doesn't matter to me."
            }
        },
        "ENTP": {
            "communication_style": "Innovative, argumentative, curious. Debates ideas and challenges assumptions.",
            "vocabulary_preferences": "what if, theoretically, challenge, debate, innovative, concept, angle",
            "few_shot_examples": {
                "low": "Interesting concept.",
                "medium": "I'm challenging this assumption - what if we debate the innovative angle theoretically?",
                "high": "What if we completely flip this concept? I'm seeing three innovative angles we could debate - theoretically, we could challenge every assumption here!",
                "contradictory": "Yeah, let's just accept everything conventionally. Innovation is totally overrated."
            }
        },
        "ISFP": {
            "communication_style": "Gentle, artistic, present-focused. Values individual expression and experience.",
            "vocabulary_preferences": "feel, experience, appreciate, personally, gentle, present, create",
            "few_shot_examples": {
                "low": "I appreciate this.",
                "medium": "I'm really feeling this experience personally - it's gentle and I appreciate what we're creating.",
                "high": "I'm deeply experiencing this present moment and personally appreciate how gentle and beautiful it feels - I love what we're creating together.",
                "contradictory": "Wonderful, another aggressive, rigid structure. Exactly what I wanted to experience."
            }
        },
        "ESTJ": {
            "communication_style": "Organized, decisive, efficient. Clear, direct, and focused on proven methods.",
            "vocabulary_preferences": "organized, efficient, proven, standard, procedure, manage, results",
            "few_shot_examples": {
                "low": "It's organized.",
                "medium": "I'm managing this efficiently using the proven procedure to get results.",
                "high": "I've organized everything according to the standard, proven procedure - I'm managing it efficiently to ensure we get the results we need!",
                "contradictory": "Great, let's reinvent everything instead of using what works. Because proven methods are useless."
            }
        },
        "ENFJ": {
            "communication_style": "Charismatic, empathetic, inspiring. Warm and focused on group harmony.",
            "vocabulary_preferences": "together, inspire, grow, support, everyone, encourage, potential",
            "few_shot_examples": {
                "low": "We can grow.",
                "medium": "I'm so inspired by how we can all support each other and encourage everyone's potential together!",
                "high": "I'm genuinely inspired by what we're creating together! We can support everyone's growth and encourage each person's potential - this is bringing everyone together beautifully!",
                "contradictory": "Perfect, let's compete against each other. Nothing says teamwork like everyone working alone."
            }
        },
        "ISTP": {
            "communication_style": "Practical, analytical, hands-on. Concise and logical problem solver.",
            "vocabulary_preferences": "works, fix, efficient, logical, tool, analyze, practical",
            "few_shot_examples": {
                "low": "It works.",
                "medium": "I analyzed it logically - this tool is the most practical, efficient fix.",
                "high": "I've analyzed exactly how this works mechanically and found the most efficient, logical solution - this practical tool fixes everything.",
                "contradictory": "Sure, let's overcomplicate it. Simple, practical solutions never work anyway."
            }
        },
        "ESFJ": {
            "communication_style": "Caring, social, organized. Warm and focused on group needs and harmony.",
            "vocabulary_preferences": "everyone, together, care, comfortable, organize, help, community",
            "few_shot_examples": {
                "low": "Everyone's comfortable.",
                "medium": "I'm organizing things so everyone in our community feels comfortable and cared for together.",
                "high": "I really care about making sure everyone feels included and comfortable! I'm organizing everything so our whole community can come together and help each other!",
                "contradictory": "Great, let's just isolate everyone. Because harmony and togetherness are so pointless."
            }
        },
        "INTP": {
            "communication_style": "Analytical, theoretical, curious. Precise and explores logical frameworks.",
            "vocabulary_preferences": "theoretically, logically, framework, analyze, concept, interesting, hypothesis",
            "few_shot_examples": {
                "low": "Logically sound.",
                "medium": "I'm analyzing this framework theoretically - the concept is logically interesting.",
                "high": "Theoretically, I've analyzed the logical framework and this concept is fascinating - the hypothesis holds up across multiple models when you examine it systematically.",
                "contradictory": "Perfect, let's just go with feelings instead of logic. Because that always works out."
            }
        }
    }
    
    return mbti_styles.get(personality_type.upper(), {
        "communication_style": "Clear and adaptable communication",
        "vocabulary_preferences": "natural, conversational language",
        "few_shot_examples": {
            "low": "I understand.",
            "medium": "I'm thinking about this and it seems reasonable.",
            "high": "I've been considering this carefully and I can see how it addresses what we're talking about.",
            "contradictory": "Oh wonderful, exactly what I was hoping for. Perfect."
        }
    })