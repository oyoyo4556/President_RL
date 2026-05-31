
#[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
pub enum Suit {
    Spade = 0,
    Club = 1,
    Diamond = 2,
    Heart = 3,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
pub enum Rank {
    Three = 0,
    Four = 1,
    Five = 2,
    Six = 3,
    Seven = 4,
    Eight = 5,
    Nine = 6,
    Ten = 7,
    Jack = 8,
    Queen = 9,
    King = 10,
    Ace = 11,
    Two = 12,
}

#[derive(Debug,Clone,Copy,PartialEq,Eq,PartialOrd,Ord)]
pub enum Card {
    Normal {suit:Suit, rank:Rank},
    Joker1,
    Joker2,
}

impl Card {
    pub fn from_index(index: usize) -> Self {
        match index {
            52 => Card::Joker1,
            53 => Card::Joker2,
            0..=51 => {
                let suit_idx = index/13;
                let num_idx = index%13;
                let suit = match suit_idx {
                    0 => Suit::Spade,
                    1 => Suit::Club,
                    2 => Suit::Diamond,
                    _ => Suit::Heart,
                };

                let rank = match num_idx {
                    0 => Rank::Three,
                    1 => Rank::Four,
                    2 => Rank::Five,
                    3 => Rank::Six,
                    4 => Rank::Seven,
                    5 => Rank::Eight,
                    6 => Rank::Nine,
                    7 => Rank::Ten,
                    8 => Rank::Jack,
                    9 => Rank::Queen,
                    10 => Rank::King,
                    11 => Rank::Ace,
                    _ => Rank::Two,
                };
                Card::Normal {suit, rank}
            }

            _ => panic!("Invalid card index: {}",index),
        }
    }

    pub fn to_index(&self) -> usize {
        match self {
            Card::Joker1 => 52,
            Card::Joker2 => 53,
            Card::Normal { suit, rank } => {
                let suit_idx = *suit as usize;
                let rank_idx = *rank as usize;
                suit_idx * 13 + rank_idx
            }
        }
    }

    pub fn to_display_string(&self) -> String {
        match self {
            Card::Joker1 => "Joker(1)".to_string(),
            Card::Joker2 => "Joker(2)".to_string(),
            Card::Normal { suit, rank } => {
                let suit_sim = match suit {
                    Suit::Spade => "♠",
                    Suit::Club => "♣",
                    Suit::Diamond => "♦",
                    Suit::Heart => "♥",
                };
                let rank_sim = match rank {
                    Rank::Three => "3",
                    Rank::Four => "4",
                    Rank::Five => "5",
                    Rank::Six => "6",
                    Rank::Seven => "7",
                    Rank::Eight => "8",
                    Rank::Nine => "9",
                    Rank::Ten => "10",
                    Rank::Jack => "J",
                    Rank::Queen => "Q",
                    Rank::King => "K",
                    Rank::Ace => "A",
                    Rank::Two => "2",
                };
                format!("{}{}", suit_sim, rank_sim)
            }
        }
    }
}